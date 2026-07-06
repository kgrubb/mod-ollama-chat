#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include "DatabaseEnv.h"
#include "Player.h"
#include <fmt/core.h>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_set>

float GetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid)
{
    if (!g_EnableSentimentTracking)
        return g_SentimentDefaultValue;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    
    auto botIt = g_BotPlayerSentiments.find(botGuid);
    if (botIt != g_BotPlayerSentiments.end())
    {
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt != botIt->second.end())
        {
            return playerIt->second;
        }
    }
    
    // Return default value if not found
    return g_SentimentDefaultValue;
}

void SetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid, float sentimentValue)
{
    if (!g_EnableSentimentTracking)
        return;

    // Clamp sentiment value to valid range [0.0, 1.0]
    sentimentValue = std::max(0.0f, std::min(1.0f, sentimentValue));
    
    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    g_BotPlayerSentiments[botGuid][playerGuid] = sentimentValue;
    
    if (g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[OllamaChat] Set sentiment between bot {} and player {} to {:.2f}", 
                 botGuid, playerGuid, sentimentValue);
    }
}

static float ParseSentimentAdjustment(std::string const& response)
{
    if (response.empty())
        return 0.0f;

    std::string upper = response;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper.find("POSITIVE") != std::string::npos)
        return g_SentimentAdjustmentStrength;
    if (upper.find("NEGATIVE") != std::string::npos)
        return -g_SentimentAdjustmentStrength;
    return 0.0f;
}

static void AdjustBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid, float adjustment)
{
    if (!g_EnableSentimentTracking || adjustment == 0.0f)
        return;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    auto& pairMap = g_BotPlayerSentiments[botGuid];
    auto it = pairMap.find(playerGuid);
    float const prev = it != pairMap.end() ? it->second : g_SentimentDefaultValue;
    float const next = std::max(0.0f, std::min(1.0f, prev + adjustment));
    pairMap[playerGuid] = next;

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] Sentiment {} -> {} ({:+.2f}) bot {} player {}",
            prev, next, adjustment, botGuid, playerGuid);
}

void UpdateBotPlayerSentiment(Player* bot, Player* player, const std::string& message)
{
    if (!g_EnableSentimentTracking || !bot || !player || message.empty())
        return;

    uint64_t const botGuid = bot->GetGUID().GetRawValue();
    uint64_t const playerGuid = player->GetGUID().GetRawValue();

    std::string prompt = SafeFormat(g_SentimentAnalysisPrompt, fmt::arg("message", message));
    auto future = SubmitQuery(prompt);
    if (!future.valid())
        return;

    std::thread([botGuid, playerGuid, future = std::move(future)]() mutable {
        try
        {
            AdjustBotPlayerSentiment(botGuid, playerGuid,
                ParseSentimentAdjustment(future.valid() ? future.get() : ""));
        }
        catch (...)
        {
        }
    }).detach();
}

std::string GetSentimentPromptAddition(Player* bot, Player* player)
{
    if (!g_EnableSentimentTracking || !bot || !player)
        return "";

    float sentimentValue = GetBotPlayerSentiment(bot->GetGUID().GetRawValue(), player->GetGUID().GetRawValue());

    return fmt::format("Relationship with {}: {:.2f} (0=hostile, 0.5=neutral, 1=friendly)",
        player->GetName(), sentimentValue) +
        (sentimentValue < 0.4f
            ? "\nRelationship is strained. You may be blunt or dismissive, but not insulting or escalating."
            : "");
}

void LoadBotPlayerSentimentsFromDB()
{
    if (!g_EnableSentimentTracking)
        return;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    g_BotPlayerSentiments.clear();
    
    QueryResult result = CharacterDatabase.Query("SELECT bot_guid, player_guid, sentiment_value FROM mod_ollama_chat_bot_player_sentiments");
    
    if (!result)
    {
        LOG_INFO("server.loading", "[OllamaChat] No existing sentiment data found in database");
        return;
    }
    
    uint32_t count = 0;
    do
    {
        Field* fields = result->Fetch();
        uint64_t botGuid = fields[0].Get<uint64_t>();
        uint64_t playerGuid = fields[1].Get<uint64_t>();
        float sentimentValue = fields[2].Get<float>();
        
        g_BotPlayerSentiments[botGuid][playerGuid] = sentimentValue;
        count++;
        
    } while (result->NextRow());
    
    LOG_INFO("server.loading", "[OllamaChat] Loaded {} sentiment records from database", count);
}

void SaveBotPlayerSentimentsToDB()
{
    if (!g_EnableSentimentTracking)
        return;

    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    
    if (g_BotPlayerSentiments.empty())
        return;
    
    // Batch all upserts into one async transaction rather than a statement per pair.
    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    for (const auto& [botGuid, playerMap] : g_BotPlayerSentiments)
    {
        for (const auto& [playerGuid, sentimentValue] : playerMap)
        {
            trans->Append(SafeFormat(
                "REPLACE INTO mod_ollama_chat_bot_player_sentiments (bot_guid, player_guid, sentiment_value) "
                "VALUES ({}, {}, {:.3f})",
                botGuid, playerGuid, sentimentValue));
        }
    }
    if (trans->GetSize() > 0)
        CharacterDatabase.CommitTransaction(trans);
    
    if (g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[OllamaChat] Saved sentiment data to database");
    }
}

void PurgeOrphanedSentiments()
{
    if (!g_EnableSentimentTracking)
        return;

    // Throttled to once per hour; random bots recycle their guid on re-randomize and
    // leave behind sentiment rows whose bot character no longer exists.
    static time_t lastPurge = 0;
    time_t now = time(nullptr);
    if (lastPurge && difftime(now, lastPurge) < 3600.0)
        return;
    lastPurge = now;

    std::vector<uint64_t> cachedBots;
    {
        std::lock_guard<std::mutex> lock(g_SentimentMutex);
        cachedBots.reserve(g_BotPlayerSentiments.size());
        for (auto const& [botGuid, playerMap] : g_BotPlayerSentiments)
            cachedBots.push_back(botGuid);
    }

    CharacterDatabase.Execute(
        "DELETE FROM mod_ollama_chat_bot_player_sentiments WHERE bot_guid NOT IN (SELECT guid FROM characters)");

    if (cachedBots.empty())
        return;

    std::ostringstream ids;
    for (size_t i = 0; i < cachedBots.size(); ++i)
        ids << (i ? "," : "") << cachedBots[i];

    std::unordered_set<uint64_t> live;
    if (QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT guid FROM characters WHERE guid IN ({})", ids.str())))
    {
        do
        {
            live.insert((*result)[0].Get<uint64_t>());
        } while (result->NextRow());
    }

    // Drop cached entries for deleted bots so the next save does not re-insert them.
    std::lock_guard<std::mutex> lock(g_SentimentMutex);
    for (uint64_t botGuid : cachedBots)
        if (!live.count(botGuid))
            g_BotPlayerSentiments.erase(botGuid);
}

void InitializeSentimentTracking()
{
    if (!g_EnableSentimentTracking)
    {
        LOG_INFO("server.loading", "[OllamaChat] Sentiment tracking is disabled");
        return;
    }
    
    LOG_INFO("server.loading", "[OllamaChat] Initializing sentiment tracking system...");
    
    // Load existing sentiment data from database
    LoadBotPlayerSentimentsFromDB();
    
    // Initialize the last save time
    g_LastSentimentSaveTime = time(nullptr);
    
    LOG_INFO("server.loading", "[OllamaChat] Sentiment tracking system initialized");
}
