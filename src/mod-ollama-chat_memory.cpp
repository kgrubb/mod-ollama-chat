#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include "DatabaseEnv.h"
#include "ObjectAccessor.h"
#include <fmt/core.h>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cctype>
#include <unordered_set>
#include <set>
#include <mutex>
#include <utility>

static bool MemoryCommit(CharacterDatabaseTransaction const& trans, char const* op)
{
    if (!trans || trans->GetSize() == 0)
        return true;

    TransactionCallback callback(CharacterDatabase.AsyncCommitTransaction(trans));
    if (!callback.m_future.get())
    {
        LOG_ERROR("server.loading", "[OllamaChat] Memory {} failed", op);
        return false;
    }
    return true;
}

static bool MemoryExecute(std::string const& sql, char const* op)
{
    if (sql.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] Memory {} failed: empty SQL", op);
        return false;
    }

    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    trans->Append(sql);
    return MemoryCommit(trans, op);
}

namespace
{
using ScoredLine = std::pair<std::string, float>;

uint64_t MemoryPairKey(uint64_t botGuid, uint64_t playerGuid)
{
    return botGuid ^ (playerGuid * 2654435761ULL);
}

uint64_t MemoryJobKey(MemoryJobType type, uint64_t botGuid, uint64_t playerGuid)
{
    return MemoryPairKey(botGuid, playerGuid) ^ (static_cast<uint64_t>(type) << 32);
}

struct MemorySections
{
    std::string profile;
    std::string history;
};

std::unordered_map<uint64_t, std::unordered_map<uint64_t, time_t>> g_CompactionCooldownUntil;
std::mutex g_CompactionCooldownMutex;

static uint32_t GetWatermark(uint64_t botGuid, uint64_t playerGuid);
static void SetWatermark(uint64_t botGuid, uint64_t playerGuid, uint32_t count);
static std::string GetSemanticText(uint64_t botGuid, uint64_t playerGuid);

static bool IsCompactionOnCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_CompactionCooldownMutex);
    auto botIt = g_CompactionCooldownUntil.find(botGuid);
    if (botIt == g_CompactionCooldownUntil.end())
        return false;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return false;
    return time(nullptr) < playerIt->second;
}

static void SetCompactionCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_CompactionCooldownMutex);
    g_CompactionCooldownUntil[botGuid][playerGuid] =
        time(nullptr) + static_cast<time_t>(OllamaMemory::CompactionFailCooldownSeconds);
}

static void ClearCompactionCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_CompactionCooldownMutex);
    auto botIt = g_CompactionCooldownUntil.find(botGuid);
    if (botIt != g_CompactionCooldownUntil.end())
        botIt->second.erase(playerGuid);
}

static void AdvanceCompactionWatermark(uint64_t botGuid, uint64_t playerGuid, uint32_t snapshotWm, uint32_t pendingCount)
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return;

    uint32_t newWm = std::min(snapshotWm + pendingCount, static_cast<uint32_t>(playerIt->second.size()));
    SetWatermark(botGuid, playerGuid, newWm);
}

static void FailCompaction(uint64_t botGuid, uint64_t playerGuid, uint32_t snapshotWm, uint32_t pendingCount,
    char const* reason)
{
    AdvanceCompactionWatermark(botGuid, playerGuid, snapshotWm, pendingCount);
    SetCompactionCooldown(botGuid, playerGuid);
    LOG_WARN("server.loading", "[OllamaChat] Memory compaction failed bot {} player {}: {}", botGuid, playerGuid, reason);
}

constexpr char kProfileMarker[] = "[PROFILE]";
constexpr char kHistoryMarker[] = "[HISTORY]";
constexpr uint32_t kCompactionPreFlushTurns = 6;

static void AppendDialogueTurn(std::ostringstream& out, std::string const& playerName,
    std::string const& botName, ConversationTurn const& turn, bool tagUnverified)
{
    out << playerName << ": " << turn.playerMessage << "\n";
    out << botName << ": " << turn.botReply;
    if (tagUnverified && !turn.verified)
        out << " (unverified)";
    out << "\n";
}

static bool HasRecallCue(std::string const& query)
{
    std::string lower = query;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    static char const* kCues[] = { "remember", "last time", "earlier", "you said", "recall" };
    for (char const* cue : kCues)
    {
        if (lower.find(cue) != std::string::npos)
            return true;
    }
    return false;
}

constexpr char kPromptTemplate[] =
    "Memory of {player_name}:\n"
    "Profile:\n{profile}\n\nRecent chat:\n{history}{past_recall}\n"
    "Use only memories listed here.";

static std::string BuildNudgeUserPrompt(std::string const& profile, std::string const& history, std::string const& recent)
{
    std::string u;
    u.reserve(profile.size() + history.size() + recent.size() + 48);
    u.append("Profile:\n");
    u.append(profile.empty() ? "(none)" : profile);
    u.append("\n\nHistory:\n");
    u.append(history.empty() ? "(none)" : history);
    u.append("\n\nRecent:\n");
    u.append(recent);
    return u;
}

static std::vector<std::string> Tokenize(const std::string& text)
{
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::vector<std::string> tokens;
    std::stringstream ss(lower);
    std::string tok;
    while (ss >> tok)
    {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::ispunct), tok.end());
        if (!tok.empty())
            tokens.push_back(tok);
    }
    return tokens;
}

static std::unordered_map<std::string, float> ToTF(const std::vector<std::string>& tokens)
{
    std::unordered_map<std::string, float> tf;
    for (const auto& t : tokens)
        tf[t] += 1.0f;
    return tf;
}

static float CosineSim(const std::unordered_map<std::string, float>& a, const std::unordered_map<std::string, float>& b)
{
    float dot = 0, na = 0, nb = 0;
    for (const auto& [k, v] : a)
    {
        na += v * v;
        auto it = b.find(k);
        if (it != b.end())
            dot += v * it->second;
    }
    for (const auto& [k, v] : b)
        nb += v * v;
    if (na == 0 || nb == 0)
        return 0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static std::vector<std::string> SplitMemoryLines(const std::string& blob)
{
    std::vector<std::string> lines;
    std::stringstream ss(blob);
    std::string line;
    while (std::getline(ss, line))
    {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            continue;
        lines.push_back(line.substr(start));
    }
    return lines;
}

static std::string RetrieveRelevantMemory(const std::string& query, const std::string& blob,
    uint32_t maxItems, float threshold, bool fallbackIfEmpty = true)
{
    auto lines = SplitMemoryLines(blob);
    if (lines.empty())
        return "";

    auto qVec = ToTF(Tokenize(query));
    std::vector<ScoredLine> scored;
    scored.reserve(lines.size());
    for (const auto& line : lines)
    {
        float s = CosineSim(qVec, ToTF(Tokenize(line)));
        if (s >= threshold)
            scored.emplace_back(line, s);
    }

    if (scored.empty())
    {
        if (!fallbackIfEmpty)
            return "";
        std::ostringstream out;
        uint32_t take = std::min(maxItems, static_cast<uint32_t>(lines.size()));
        for (uint32_t i = 0; i < take; ++i)
            out << lines[i] << "\n";
        std::string r = out.str();
        while (!r.empty() && (r.back() == '\n' || r.back() == '\r'))
            r.pop_back();
        return r;
    }

    if (scored.size() > maxItems)
    {
        std::partial_sort(
            scored.begin(), scored.begin() + maxItems, scored.end(),
            [](ScoredLine const& a, ScoredLine const& b) { return a.second > b.second; });
        scored.resize(maxItems);
    }
    else
    {
        std::sort(scored.begin(), scored.end(), [](ScoredLine const& a, ScoredLine const& b) { return a.second > b.second; });
    }

    std::ostringstream out;
    for (auto const& sl : scored)
        out << sl.first << "\n";
    std::string r = out.str();
    while (!r.empty() && (r.back() == '\n' || r.back() == '\r'))
        r.pop_back();
    return r;
}

static std::string TruncateMemory(const std::string& text, uint32_t maxChars)
{
    if (text.size() <= maxChars)
        return text;
    return text.substr(0, maxChars);
}

static MemorySections ParseMemorySections(std::string const& blob)
{
    MemorySections sec;
    size_t pPos = blob.find(kProfileMarker);
    size_t hPos = blob.find(kHistoryMarker);
    if (pPos == std::string::npos && hPos == std::string::npos)
    {
        sec.history = blob;
        return sec;
    }
    if (pPos != std::string::npos)
    {
        size_t start = pPos + sizeof(kProfileMarker) - 1;
        size_t end = hPos != std::string::npos && hPos > pPos ? hPos : blob.size();
        sec.profile = blob.substr(start, end - start);
        size_t trim = sec.profile.find_first_not_of(" \t\r\n");
        if (trim != std::string::npos)
            sec.profile = sec.profile.substr(trim);
        while (!sec.profile.empty() && (sec.profile.back() == '\n' || sec.profile.back() == '\r'))
            sec.profile.pop_back();
    }
    if (hPos != std::string::npos)
    {
        size_t start = hPos + sizeof(kHistoryMarker) - 1;
        sec.history = blob.substr(start);
        size_t trim = sec.history.find_first_not_of(" \t\r\n");
        if (trim != std::string::npos)
            sec.history = sec.history.substr(trim);
    }
    return sec;
}

static std::string FormatMemorySections(std::string const& profile, std::string const& history)
{
    std::ostringstream out;
    if (!profile.empty())
        out << kProfileMarker << "\n" << profile;
    if (!history.empty())
    {
        if (!profile.empty())
            out << "\n";
        out << kHistoryMarker << "\n" << history;
    }
    return out.str();
}

static std::string TrimMemoryResponse(std::string s)
{
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

static void FillEmptyMemorySections(MemorySections& out, MemorySections const& keep)
{
    if (out.profile.empty())
        out.profile = keep.profile;
    if (out.history.empty())
        out.history = keep.history;
}

static std::string BuildArchiveRecall(uint64_t botGuid, uint64_t playerGuid, std::string const& query)
{
    std::ostringstream candidates;
    bool queryDb = false;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (!GetSemanticText(botGuid, playerGuid).empty())
            queryDb = true;

        bool hasPending = false;
        auto apIt = g_ArchivePending.find(botGuid);
        if (apIt != g_ArchivePending.end())
        {
            auto pit = apIt->second.find(playerGuid);
            if (pit != apIt->second.end() && !pit->second.empty())
            {
                hasPending = true;
                queryDb = true;
                for (auto const& turn : pit->second)
                    candidates << "Player: " << turn.first << "\nBot: " << turn.second << "\n";
            }
        }

        auto histIt = g_BotConversationHistory.find(botGuid);
        if (histIt != g_BotConversationHistory.end())
        {
            auto pit = histIt->second.find(playerGuid);
            if (pit != histIt->second.end() && !pit->second.empty())
            {
                queryDb = true;
                if (!hasPending)
                {
                    uint32_t wm = GetWatermark(botGuid, playerGuid);
                    for (size_t i = wm; i < pit->second.size(); ++i)
                        candidates << "Player: " << pit->second[i].playerMessage
                                   << "\nBot: " << pit->second[i].botReply << "\n";
                }
            }
        }
    }

    if (!queryDb)
        return "";

    if (QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT context, bot_reply FROM mod_ollama_chat_memory_archive "
            "WHERE bot_guid = {} AND player_guid = {} ORDER BY created_at DESC LIMIT {}",
            botGuid, playerGuid, OllamaMemory::ArchiveRecallWindow)))
    {
        do
        {
            candidates << "Player: " << (*result)[0].Get<std::string>()
                       << "\nBot: " << (*result)[1].Get<std::string>() << "\n";
        } while (result->NextRow());
    }

    std::string blob = candidates.str();
    if (blob.empty())
        return "";
    std::string recalled = RetrieveRelevantMemory(
        query, blob, OllamaMemory::RecallMaxItems, OllamaMemory::RecallThreshold, false);
    if (recalled.empty())
        return "";
    return "\n\nRelevant past moments:\n" + recalled;
}

static bool DequeueMemoryJob(MemoryJobType type, MemoryJob& job)
{
    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    auto it = std::find_if(g_MemoryCompactionQueue.begin(), g_MemoryCompactionQueue.end(),
        [type](MemoryJob const& j) { return j.type == type; });
    if (it == g_MemoryCompactionQueue.end())
        return false;
    job = *it;
    g_MemoryCompactionQueue.erase(it);
    return true;
}

static bool IsTrivialTurn(std::string const& msg)
{
    size_t start = msg.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return true;
    std::string s = msg.substr(start);
    size_t end = s.find_last_not_of(" \t\r\n");
    if (end != std::string::npos)
        s = s.substr(0, end + 1);
    if (s.find(' ') == std::string::npos && s.size() < 20)
    {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "brb" || lower == "lfm" || lower == "gtg" || lower == "ok" || lower == "ty")
            return true;
        return s.size() < 8;
    }
    return false;
}

static uint32_t GetWatermark(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_CompactedTurnCount.find(botGuid);
    if (botIt == g_CompactedTurnCount.end())
        return 0;
    auto playerIt = botIt->second.find(playerGuid);
    return playerIt == botIt->second.end() ? 0 : playerIt->second;
}

static void SetWatermark(uint64_t botGuid, uint64_t playerGuid, uint32_t count)
{
    g_CompactedTurnCount[botGuid][playerGuid] = count;
}

static std::string GetSemanticText(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_SemanticMemory.find(botGuid);
    if (botIt == g_SemanticMemory.end())
        return "";
    auto playerIt = botIt->second.find(playerGuid);
    return playerIt == botIt->second.end() ? "" : playerIt->second;
}

static void SetSemanticText(uint64_t botGuid, uint64_t playerGuid, const std::string& text)
{
    g_SemanticMemory[botGuid][playerGuid] = text;
}

static bool ShouldCompactUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return false;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return false;

    size_t dequeSize = playerIt->second.size();
    uint32_t wm = GetWatermark(botGuid, playerGuid);
    if (dequeSize <= wm)
        return false;

    uint32_t pending = static_cast<uint32_t>(dequeSize - wm);
    if (pending >= OllamaMemory::CompactionThreshold)
        return true;
    if (dequeSize >= OllamaMemory::DequeCap)
        return true;

    auto tsIt = g_TurnTimestamps.find(botGuid);
    if (tsIt != g_TurnTimestamps.end())
    {
        auto ptsIt = tsIt->second.find(playerGuid);
        if (ptsIt != tsIt->second.end() && ptsIt->second.size() > wm)
        {
            time_t oldest = ptsIt->second[wm];
            time_t now = time(nullptr);
            double ageSec = difftime(now, oldest);
            if (OllamaMemory::CompactionMaxAgeMinutes > 0 &&
                ageSec >= static_cast<double>(OllamaMemory::CompactionMaxAgeMinutes) * 60)
                return true;
            if (OllamaMemory::CompactionMaxAgeMinutes == 0 && OllamaMemory::EpisodicMaxAgeDays > 0 &&
                ageSec >= static_cast<double>(OllamaMemory::EpisodicMaxAgeDays) * 86400)
                return true;
        }
    }
    return false;
}

static void RunCompaction(uint64_t botGuid, uint64_t playerGuid)
{
    std::vector<ConversationTurn> turns;
    std::string existingMemory;
    uint32_t pendingCount = 0;
    uint32_t snapshotWm = 0;
    float sentiment = g_SentimentDefaultValue;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return;
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt == botIt->second.end())
            return;

        snapshotWm = GetWatermark(botGuid, playerGuid);
        size_t dequeSize = playerIt->second.size();
        if (dequeSize <= snapshotWm)
            return;

        pendingCount = static_cast<uint32_t>(dequeSize - snapshotWm);
        size_t flushStart = dequeSize > kCompactionPreFlushTurns ? dequeSize - kCompactionPreFlushTurns : 0;
        size_t blockStart = std::min(static_cast<size_t>(snapshotWm), flushStart);
        turns.assign(playerIt->second.begin() + static_cast<std::ptrdiff_t>(blockStart), playerIt->second.end());
        existingMemory = GetSemanticText(botGuid, playerGuid);
    }

    Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
    Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
    std::string playerName = player ? player->GetName() : "Player";
    std::string botName = bot ? bot->GetName() : "Bot";

    if (bot && g_EnableSentimentTracking)
        sentiment = GetBotPlayerSentiment(botGuid, playerGuid);

    std::ostringstream verifiedBlock;
    std::ostringstream unverifiedBlock;
    for (ConversationTurn const& turn : turns)
    {
        if (turn.verified)
            AppendDialogueTurn(verifiedBlock, playerName, botName, turn, false);
        else
            AppendDialogueTurn(unverifiedBlock, playerName, botName, turn, true);
    }

    BotContext ctx;
    ScenarioInput input;
    input.compactionExistingMemory = existingMemory.empty() ? "(none)" : existingMemory;
    input.compactionEpisodicVerified = verifiedBlock.str();
    input.compactionEpisodicUnverified = unverifiedBlock.str();
    input.compactionPlayerName = playerName;
    input.compactionSentiment = sentiment;

    PromptBundle bundle = OllamaPromptComposer::Build(PromptScenario::MemoryCompaction, ctx, input);
    bundle.maxTokens = OllamaMemory::MemoryQueryMaxTokens;
    auto future = SubmitQuery(bundle);
    std::string response = future.valid() ? future.get() : "";
    if (response.empty())
    {
        FailCompaction(botGuid, playerGuid, snapshotWm, pendingCount, "empty LLM response");
        return;
    }

    MemorySections existing = ParseMemorySections(existingMemory);
    MemorySections sections = ParseMemorySections(TrimMemoryResponse(response));
    FillEmptyMemorySections(sections, existing);
    if (sections.profile.empty() && sections.history.empty())
    {
        FailCompaction(botGuid, playerGuid, snapshotWm, pendingCount, "empty memory");
        return;
    }

    ClearCompactionCooldown(botGuid, playerGuid);

    std::string newMemory = TruncateMemory(
        FormatMemorySections(sections.profile, sections.history), OllamaMemory::MaxSemanticChars);

    uint32_t newWm = 0;
    bool deleteEpisodic = false;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return;
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt == botIt->second.end() || playerIt->second.size() < snapshotWm)
            return;

        SetSemanticText(botGuid, playerGuid, newMemory);
        uint32_t currentWm = GetWatermark(botGuid, playerGuid);
        newWm = std::min(currentWm + pendingCount, static_cast<uint32_t>(playerIt->second.size()));
        SetWatermark(botGuid, playerGuid, newWm);

        auto& hist = playerIt->second;
        while (hist.size() > OllamaMemory::DequeCap && GetWatermark(botGuid, playerGuid) > 0)
        {
            hist.pop_front();
            SetWatermark(botGuid, playerGuid, GetWatermark(botGuid, playerGuid) - 1);
            auto& ts = g_TurnTimestamps[botGuid][playerGuid];
            if (!ts.empty())
                ts.pop_front();
        }

        deleteEpisodic = g_PersistEpisodicMemory;
    }

    std::string escMem = newMemory;
    CharacterDatabase.EscapeString(escMem);
    std::string escName = playerName;
    CharacterDatabase.EscapeString(escName);

    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    if (deleteEpisodic)
    {
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memory_episodic WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid));
    }
    trans->Append(SafeFormat(
        "REPLACE INTO mod_ollama_chat_memory_semantic (bot_guid, player_guid, player_name, memory_text, "
        "compacted_turn_count, last_compacted_at) VALUES ({}, {}, '{}', '{}', {}, NOW())",
        botGuid, playerGuid, escName, escMem, newWm));
    MemoryCommit(trans, "compaction");

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] Compacted memory for bot {} player {} ({} turns)",
                 botGuid, playerGuid, pendingCount);
}

static void RunNudge(uint64_t botGuid, uint64_t playerGuid)
{
    std::string profile;
    std::string history;
    std::string recent;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        time_t now = time(nullptr);
        auto nudgeIt = g_LastNudgeTime.find(botGuid);
        if (nudgeIt != g_LastNudgeTime.end())
        {
            auto lastIt = nudgeIt->second.find(playerGuid);
            if (lastIt != nudgeIt->second.end() &&
                difftime(now, lastIt->second) < static_cast<double>(OllamaMemory::NudgeMinIntervalSeconds))
                return;
        }

        MemorySections sec = ParseMemorySections(GetSemanticText(botGuid, playerGuid));
        profile = sec.profile;
        history = sec.history;

        auto histIt = g_BotConversationHistory.find(botGuid);
        if (histIt == g_BotConversationHistory.end())
            return;
        auto pairIt = histIt->second.find(playerGuid);
        if (pairIt == histIt->second.end() || pairIt->second.empty())
            return;

        size_t from = pairIt->second.size() > 3 ? pairIt->second.size() - 3 : 0;
        std::ostringstream ep;
        for (size_t i = from; i < pairIt->second.size(); ++i)
            ep << "Player: " << pairIt->second[i].playerMessage << "\nBot: " << pairIt->second[i].botReply << "\n";
        recent = ep.str();
        if (recent.empty())
            return;
    }

    PromptBundle bundle;
    bundle.system = GetMemoryNudgeSystemPrompt();
    bundle.user = BuildNudgeUserPrompt(profile, history, recent);
    bundle.maxTokens = OllamaMemory::MemoryQueryMaxTokens;
    auto future = SubmitQuery(bundle);
    std::string response = future.valid() ? future.get() : "";
    if (response.empty())
        return;

    MemorySections existing{profile, history};
    MemorySections updated = ParseMemorySections(TrimMemoryResponse(response));
    FillEmptyMemorySections(updated, existing);
    if (updated.profile.empty() && updated.history.empty())
        return;

    std::string oldMemory = FormatMemorySections(profile, history);
    std::string newMemory = TruncateMemory(
        FormatMemorySections(updated.profile, updated.history), OllamaMemory::MaxSemanticChars);
    if (newMemory == oldMemory)
        return;

    uint32_t wm = 0;
    std::string playerName = "Player";
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        SetSemanticText(botGuid, playerGuid, newMemory);
        g_LastNudgeTime[botGuid][playerGuid] = time(nullptr);
        wm = GetWatermark(botGuid, playerGuid);
    }

    Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
    if (player)
        playerName = player->GetName();

    std::string escMem = newMemory;
    std::string escName = playerName;
    CharacterDatabase.EscapeString(escMem);
    CharacterDatabase.EscapeString(escName);
    MemoryExecute(SafeFormat(
        "REPLACE INTO mod_ollama_chat_memory_semantic (bot_guid, player_guid, player_name, memory_text, "
        "compacted_turn_count, last_compacted_at) VALUES ({}, {}, '{}', '{}', {}, NOW())",
        botGuid, playerGuid, escName, escMem, wm), "nudge");
}

static void DispatchMemoryJob(MemoryJob const& job)
{
    uint64_t jobKey = MemoryJobKey(job.type, job.botGuid, job.playerGuid);
    ++g_MemoryCompactionInFlight;
    std::thread([job, jobKey]() {
        if (job.type == MemoryJobType::Compact)
            RunCompaction(job.botGuid, job.playerGuid);
        else
            RunNudge(job.botGuid, job.playerGuid);
        --g_MemoryCompactionInFlight;
        std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
        g_MemoryCompactionPending.erase(jobKey);
    }).detach();
}

} // namespace

static void LoadBotMemoryFromDB();

static void PurgeStaleMemoryRows()
{
    if (OllamaMemory::EpisodicMaxAgeDays == 0)
        return;

    static time_t lastPurge = 0;
    time_t now = time(nullptr);
    if (lastPurge && difftime(now, lastPurge) < 3600.0)
        return;
    lastPurge = now;

    uint32_t days = OllamaMemory::EpisodicMaxAgeDays;
    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    trans->Append(SafeFormat(
        "DELETE FROM mod_ollama_chat_memory_episodic WHERE created_at < DATE_SUB(NOW(), INTERVAL {} DAY)", days));
    trans->Append(SafeFormat(
        "DELETE FROM mod_ollama_chat_memory_archive WHERE created_at < DATE_SUB(NOW(), INTERVAL {} DAY)", days));
    MemoryCommit(trans, "purge stale");
}

static void TrimArchivePair(uint64_t botGuid, uint64_t playerGuid)
{
    MemoryExecute(SafeFormat(
        "DELETE FROM mod_ollama_chat_memory_archive "
        "WHERE bot_guid = {} AND player_guid = {} AND id NOT IN ("
        "  SELECT id FROM ("
        "    SELECT id FROM mod_ollama_chat_memory_archive "
        "    WHERE bot_guid = {} AND player_guid = {} "
        "    ORDER BY created_at DESC LIMIT {}"
        "  ) t"
        ")",
        botGuid, playerGuid, botGuid, playerGuid, OllamaMemory::ArchiveMaxRowsPerPair), "archive trim");
}

static void InsertArchiveTurnsToDB(uint64_t botGuid, uint64_t playerGuid,
    std::vector<std::pair<std::string, std::string>> const& turns)
{
    if (turns.empty())
        return;

    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    for (auto const& turn : turns)
    {
        std::string escCtx = turn.first;
        std::string escReply = turn.second;
        CharacterDatabase.EscapeString(escCtx);
        CharacterDatabase.EscapeString(escReply);
        trans->Append(SafeFormat(
            "INSERT INTO mod_ollama_chat_memory_archive (bot_guid, player_guid, context, bot_reply) "
            "VALUES ({}, {}, '{}', '{}')",
            botGuid, playerGuid, escCtx, escReply));
    }
    if (!MemoryCommit(trans, "archive insert"))
        return;
    TrimArchivePair(botGuid, playerGuid);
}

static bool PopArchivePendingForDequeEviction(uint64_t botGuid, uint64_t playerGuid, size_t dequeSize,
    std::pair<std::string, std::string>& out)
{
    auto botIt = g_ArchivePending.find(botGuid);
    if (botIt == g_ArchivePending.end())
        return false;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end() || playerIt->second.size() < dequeSize)
        return false;

    auto& pending = playerIt->second;
    size_t const idx = pending.size() - dequeSize;
    out = pending[idx];
    pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(idx));
    if (pending.empty())
        botIt->second.erase(playerGuid);
    return true;
}

static void WaitForMemoryWorkers()
{
    for (uint32_t i = 0; i < 300 && g_MemoryCompactionInFlight.load() > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

static void WaitForMemoryWorkersBlocking()
{
    uint32_t i = 0;
    for (; i < 1200 && g_MemoryCompactionInFlight.load() > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (i >= 1200 && g_MemoryCompactionInFlight.load() > 0)
        LOG_WARN("server.loading", "[OllamaChat] Memory worker wait timed out ({} in flight)",
            g_MemoryCompactionInFlight.load());
}

static void DrainMemoryWorkersAndQueue()
{
    WaitForMemoryWorkersBlocking();
    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    g_MemoryCompactionQueue.clear();
    g_MemoryCompactionPending.clear();
}

void InitializeBotMemory()
{
    if (!g_EnableMemory)
        return;
    DrainMemoryWorkersAndQueue();
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        g_ArchivePending.clear();
        g_LastNudgeTime.clear();
        {
            std::lock_guard<std::mutex> cooldownLock(g_CompactionCooldownMutex);
            g_CompactionCooldownUntil.clear();
        }
    }
    LoadBotMemoryFromDB();
    g_LastMemorySaveTime = time(nullptr);
    LOG_INFO("server.loading", "[OllamaChat] Bot memory system initialized");
}

static void LoadBotMemoryFromDB()
{
    if (!g_EnableMemory)
        return;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        g_SemanticMemory.clear();
        g_CompactedTurnCount.clear();
        g_TurnTimestamps.clear();
        g_BotConversationHistory.clear();

        if (QueryResult result = CharacterDatabase.Query(
                "SELECT bot_guid, player_guid, player_name, memory_text, compacted_turn_count "
                "FROM mod_ollama_chat_memory_semantic"))
        {
            do
            {
                uint64_t botGuid = (*result)[0].Get<uint64_t>();
                uint64_t playerGuid = (*result)[1].Get<uint64_t>();
                std::string memText = (*result)[3].Get<std::string>();
                g_SemanticMemory[botGuid][playerGuid] = memText;
                g_CompactedTurnCount[botGuid][playerGuid] = (*result)[4].Get<uint32_t>();
            } while (result->NextRow());
        }

        if (g_PersistEpisodicMemory)
        {
            std::unordered_set<uint64_t> rebuiltPairs;
            if (QueryResult epResult = CharacterDatabase.Query(
                    "SELECT bot_guid, player_guid, context, bot_reply, UNIX_TIMESTAMP(created_at) "
                    "FROM mod_ollama_chat_memory_episodic ORDER BY created_at ASC"))
            {
                do
                {
                    uint64_t botGuid = (*epResult)[0].Get<uint64_t>();
                    uint64_t playerGuid = (*epResult)[1].Get<uint64_t>();
                    uint64_t pairKey = MemoryPairKey(botGuid, playerGuid);
                    if (rebuiltPairs.insert(pairKey).second)
                    {
                        g_BotConversationHistory[botGuid][playerGuid].clear();
                        g_TurnTimestamps[botGuid][playerGuid].clear();
                    }

                    std::string ctx = (*epResult)[2].Get<std::string>();
                    std::string reply = (*epResult)[3].Get<std::string>();
                    time_t ts = (*epResult)[4].IsNull() ? time(nullptr) : static_cast<time_t>((*epResult)[4].Get<uint64_t>());
                    g_BotConversationHistory[botGuid][playerGuid].push_back({ ctx, reply, true });
                    g_TurnTimestamps[botGuid][playerGuid].push_back(ts);

                    auto& hist = g_BotConversationHistory[botGuid][playerGuid];
                    while (hist.size() > OllamaMemory::DequeCap)
                    {
                        hist.pop_front();
                        g_TurnTimestamps[botGuid][playerGuid].pop_front();
                    }
                } while (epResult->NextRow());
            }

            for (auto const& [botGuid, playerMap] : g_BotConversationHistory)
            {
                for (auto const& [playerGuid, hist] : playerMap)
                {
                    if (!hist.empty())
                        g_CompactedTurnCount[botGuid][playerGuid] = 0;
                }
            }
        }

        // Legacy table backfill for pairs without episodic rows.
        if (QueryResult legacy = CharacterDatabase.Query(
                "SELECT bot_guid, player_guid, player_message, bot_reply, UNIX_TIMESTAMP(timestamp) "
                "FROM mod_ollama_chat_history ORDER BY timestamp ASC"))
        {
            do
            {
                uint64_t botGuid = (*legacy)[0].Get<uint64_t>();
                uint64_t playerGuid = (*legacy)[1].Get<uint64_t>();
                auto& hist = g_BotConversationHistory[botGuid][playerGuid];
                if (!hist.empty())
                    continue;

                std::string playerMsg = (*legacy)[2].Get<std::string>();
                std::string botReply = (*legacy)[3].Get<std::string>();
                time_t ts = (*legacy)[4].IsNull() ? time(nullptr) : static_cast<time_t>((*legacy)[4].Get<uint64_t>());
                hist.push_back({ playerMsg, botReply, true });
                g_TurnTimestamps[botGuid][playerGuid].push_back(ts);
                g_CompactedTurnCount[botGuid][playerGuid] = 0;

                while (hist.size() > OllamaMemory::DequeCap)
                {
                    hist.pop_front();
                    g_TurnTimestamps[botGuid][playerGuid].pop_front();
                }
            } while (legacy->NextRow());
        }

        for (auto& [botGuid, playerMap] : g_CompactedTurnCount)
        {
            for (auto& [playerGuid, wm] : playerMap)
            {
                size_t sz = 0;
                auto histIt = g_BotConversationHistory.find(botGuid);
                if (histIt != g_BotConversationHistory.end())
                {
                    auto pit = histIt->second.find(playerGuid);
                    if (pit != histIt->second.end())
                        sz = pit->second.size();
                }
                if (wm > sz)
                    wm = static_cast<uint32_t>(sz);
            }
        }
    }

    MemoryExecute(
        "INSERT INTO mod_ollama_chat_memory_archive (bot_guid, player_guid, context, bot_reply, created_at) "
        "SELECT h.bot_guid, h.player_guid, h.player_message, h.bot_reply, h.timestamp "
        "FROM mod_ollama_chat_history h "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM mod_ollama_chat_memory_archive a "
        "  WHERE a.bot_guid = h.bot_guid AND a.player_guid = h.player_guid LIMIT 1"
        ")", "legacy archive backfill");
}

bool SaveBotMemoryToDB(bool blocking)
{
    if (!g_EnableMemory)
        return false;
    if (blocking)
        WaitForMemoryWorkersBlocking();
    else
        WaitForMemoryWorkers();

    struct SemanticSaveRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::string memText;
        uint32_t watermark;
    };

    struct EpisodicSaveRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::vector<std::pair<std::string, std::string>> pendingTurns;
    };

    struct ArchiveSaveRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::vector<std::pair<std::string, std::string>> turns;
    };

    std::vector<SemanticSaveRow> semanticRows;
    std::vector<EpisodicSaveRow> episodicRows;
    std::vector<ArchiveSaveRow> archiveRows;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

        for (auto const& [botGuid, playerMap] : g_SemanticMemory)
        {
            for (auto const& [playerGuid, memText] : playerMap)
            {
                semanticRows.push_back({
                    botGuid,
                    playerGuid,
                    memText,
                    GetWatermark(botGuid, playerGuid)
                });
            }
        }

        if (g_PersistEpisodicMemory)
        {
            for (auto const& [botGuid, playerMap] : g_BotConversationHistory)
            {
                for (auto const& [playerGuid, hist] : playerMap)
                {
                    uint32_t wm = GetWatermark(botGuid, playerGuid);
                    if (wm >= hist.size())
                        continue;

                    EpisodicSaveRow row;
                    row.botGuid = botGuid;
                    row.playerGuid = playerGuid;
                    row.pendingTurns.reserve(hist.size() - wm);
                    for (size_t i = wm; i < hist.size(); ++i)
                        row.pendingTurns.push_back({ hist[i].playerMessage, hist[i].botReply });
                    episodicRows.push_back(std::move(row));
                }
            }
        }

        for (auto const& [botGuid, playerMap] : g_ArchivePending)
        {
            for (auto const& [playerGuid, turns] : playerMap)
            {
                if (turns.empty())
                    continue;
                archiveRows.push_back({ botGuid, playerGuid, turns });
            }
        }
    }

    for (auto const& row : semanticRows)
    {
        std::string escMem = row.memText;
        CharacterDatabase.EscapeString(escMem);
        MemoryExecute(SafeFormat(
            "UPDATE mod_ollama_chat_memory_semantic SET memory_text = '{}', compacted_turn_count = {} "
            "WHERE bot_guid = {} AND player_guid = {}",
            escMem, row.watermark, row.botGuid, row.playerGuid), "save semantic");
    }

    for (auto const& row : episodicRows)
    {
        CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memory_episodic WHERE bot_guid = {} AND player_guid = {}",
            row.botGuid, row.playerGuid));

        std::string escName = GetStoredPlayerName(row.botGuid, row.playerGuid);
        CharacterDatabase.EscapeString(escName);
        for (auto const& turn : row.pendingTurns)
        {
            std::string escCtx = turn.first;
            std::string escReply = turn.second;
            CharacterDatabase.EscapeString(escCtx);
            CharacterDatabase.EscapeString(escReply);
            trans->Append(SafeFormat(
                "INSERT INTO mod_ollama_chat_memory_episodic (bot_guid, player_guid, player_name, context, bot_reply) "
                "VALUES ({}, {}, '{}', '{}', '{}')",
                row.botGuid, row.playerGuid, escName, escCtx, escReply));
        }
        MemoryCommit(trans, "save episodic");
    }

    std::vector<std::pair<uint64_t, uint64_t>> flushedArchive;
    for (auto const& row : archiveRows)
    {
        InsertArchiveTurnsToDB(row.botGuid, row.playerGuid, row.turns);
        flushedArchive.emplace_back(row.botGuid, row.playerGuid);
    }
    if (!flushedArchive.empty())
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        for (auto const& [botGuid, playerGuid] : flushedArchive)
        {
            auto botIt = g_ArchivePending.find(botGuid);
            if (botIt != g_ArchivePending.end())
                botIt->second.erase(playerGuid);
        }
    }

    std::set<std::pair<uint64_t, uint64_t>> trimPairs;
    for (auto const& row : semanticRows)
        trimPairs.emplace(row.botGuid, row.playerGuid);
    for (auto const& row : episodicRows)
        trimPairs.emplace(row.botGuid, row.playerGuid);
    for (auto const& row : archiveRows)
        trimPairs.emplace(row.botGuid, row.playerGuid);
    for (auto const& [botGuid, playerGuid] : trimPairs)
        TrimArchivePair(botGuid, playerGuid);
    return true;
}

MemoryCompactionEnqueueResult EnqueueMemoryCompaction(uint64_t botGuid, uint64_t playerGuid, bool force)
{
    if (!g_EnableMemory)
        return MemoryCompactionEnqueueResult::NothingToCompact;

    uint64_t key = MemoryJobKey(MemoryJobType::Compact, botGuid, playerGuid);

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (force)
        {
            auto botIt = g_BotConversationHistory.find(botGuid);
            uint32_t pending = 0;
            if (botIt != g_BotConversationHistory.end())
            {
                auto playerIt = botIt->second.find(playerGuid);
                if (playerIt != botIt->second.end())
                {
                    uint32_t wm = GetWatermark(botGuid, playerGuid);
                    if (playerIt->second.size() > wm)
                        pending = static_cast<uint32_t>(playerIt->second.size() - wm);
                }
            }
            if (pending == 0)
            {
                LOG_WARN("server.loading", "[OllamaChat] Force compact skipped: no pending turns bot {} player {}",
                    botGuid, playerGuid);
                return MemoryCompactionEnqueueResult::NothingToCompact;
            }
        }
        else if (!ShouldCompactUnlocked(botGuid, playerGuid))
            return MemoryCompactionEnqueueResult::NothingToCompact;
    }

    if (!force && IsCompactionOnCooldown(botGuid, playerGuid))
        return MemoryCompactionEnqueueResult::NothingToCompact;

    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    if (g_MemoryCompactionQueue.size() >= OllamaMemory::MaxCompactionQueue)
        return MemoryCompactionEnqueueResult::NothingToCompact;
    if (g_MemoryCompactionPending.count(key))
        return MemoryCompactionEnqueueResult::AlreadyPending;
    g_MemoryCompactionPending.insert(key);
    g_MemoryCompactionQueue.push_back({ MemoryJobType::Compact, botGuid, playerGuid });
    return MemoryCompactionEnqueueResult::Enqueued;
}

void MaybeEnqueueMemoryCompaction(uint64_t botGuid, uint64_t playerGuid)
{
    EnqueueMemoryCompaction(botGuid, playerGuid, false);
}

void MaybeEnqueueMemoryNudge(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage, bool isEvent)
{
    if (!g_EnableMemory)
        return;
    if (!isEvent && IsTrivialTurn(playerMessage))
        return;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        time_t now = time(nullptr);
        auto nudgeIt = g_LastNudgeTime.find(botGuid);
        if (nudgeIt != g_LastNudgeTime.end())
        {
            auto lastIt = nudgeIt->second.find(playerGuid);
            if (lastIt != nudgeIt->second.end() &&
                difftime(now, lastIt->second) < static_cast<double>(OllamaMemory::NudgeMinIntervalSeconds))
                return;
        }
    }

    uint64_t key = MemoryJobKey(MemoryJobType::Nudge, botGuid, playerGuid);
    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    if (g_MemoryCompactionQueue.size() >= OllamaMemory::MaxCompactionQueue)
        return;
    if (g_MemoryCompactionPending.count(key))
        return;
    g_MemoryCompactionPending.insert(key);
    g_MemoryCompactionQueue.push_back({ MemoryJobType::Nudge, botGuid, playerGuid });
}

void AppendBotMemoryTurn(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage,
    std::string const& botReply, bool isEvent, bool verified)
{
    std::vector<std::pair<std::string, std::string>> toFlush;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto& hist = g_BotConversationHistory[botGuid][playerGuid];
        hist.push_back({ playerMessage, botReply, verified });
        g_TurnTimestamps[botGuid][playerGuid].push_back(time(nullptr));
        g_ArchivePending[botGuid][playerGuid].push_back({ playerMessage, botReply });

        while (hist.size() > OllamaMemory::DequeCap)
        {
            std::pair<std::string, std::string> turn;
            if (PopArchivePendingForDequeEviction(botGuid, playerGuid, hist.size(), turn))
                toFlush.push_back(std::move(turn));

            auto& ts = g_TurnTimestamps[botGuid][playerGuid];
            if (!ts.empty())
                ts.pop_front();
            uint32_t& wm = g_CompactedTurnCount[botGuid][playerGuid];
            if (wm > 0)
                --wm;
            hist.pop_front();
        }

        auto& pending = g_ArchivePending[botGuid][playerGuid];
        while (pending.size() > OllamaMemory::ArchivePendingCap)
        {
            toFlush.push_back(pending.front());
            pending.erase(pending.begin());
        }
        if (pending.empty())
            g_ArchivePending[botGuid].erase(playerGuid);
    }

    InsertArchiveTurnsToDB(botGuid, playerGuid, toFlush);
    MaybeEnqueueMemoryCompaction(botGuid, playerGuid);
    MaybeEnqueueMemoryNudge(botGuid, playerGuid, playerMessage, isEvent);
}

void ProcessMemoryCompactionTick()
{
    if (!g_EnableMemory)
        return;

    PurgeStaleMemoryRows();

    uint32_t toProcess = OllamaMemory::CompactionsPerTick;
    while (toProcess > 0)
    {
        MemoryJob job{};
        if (!DequeueMemoryJob(MemoryJobType::Compact, job))
            break;

        if (OllamaMemory::MaxConcurrentCompactions > 0 &&
            g_MemoryCompactionInFlight.load() >= OllamaMemory::MaxConcurrentCompactions)
        {
            std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
            g_MemoryCompactionQueue.push_front(job);
            break;
        }

        DispatchMemoryJob(job);
        --toProcess;
    }

    if (g_MemoryCompactionInFlight.load() == 0)
    {
        MemoryJob nudgeJob{};
        if (DequeueMemoryJob(MemoryJobType::Nudge, nudgeJob))
            DispatchMemoryJob(nudgeJob);
    }

    static time_t lastAgeSweep = 0;
    time_t now = time(nullptr);
    if (lastAgeSweep && difftime(now, lastAgeSweep) < static_cast<double>(OllamaMemory::CompactionAgeSweepIntervalSeconds))
        return;
    lastAgeSweep = now;

    std::vector<std::pair<uint64_t, uint64_t>> ageSweep;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        for (auto const& [botGuid, playerMap] : g_BotConversationHistory)
        {
            for (auto const& [playerGuid, hist] : playerMap)
            {
                if (ShouldCompactUnlocked(botGuid, playerGuid))
                    ageSweep.emplace_back(botGuid, playerGuid);
                if (ageSweep.size() >= 2)
                    break;
            }
            if (ageSweep.size() >= 2)
                break;
        }
    }
    for (auto const& [botGuid, playerGuid] : ageSweep)
        MaybeEnqueueMemoryCompaction(botGuid, playerGuid);
}

std::string GetMemoryPromptAddition(uint64_t botGuid, uint64_t playerGuid, const std::string& query, const std::string& playerName)
{
    if (!g_EnableMemory)
        return "";

    std::string blob;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        blob = GetSemanticText(botGuid, playerGuid);
    }

    MemorySections sections = ParseMemorySections(blob);
    if (sections.profile.empty() && sections.history.empty())
        return "";

    bool const recallCue = HasRecallCue(query);
    if (!recallCue && sections.history.empty())
        return "";

    std::string profile = sections.profile;
    std::string history = sections.history;

    if (!sections.history.empty() && OllamaMemory::RecallMaxItems > 0 &&
        (recallCue ? sections.history.size() > 400 : true))
    {
        std::string recalled = RetrieveRelevantMemory(
            query, sections.history, OllamaMemory::RecallMaxItems, OllamaMemory::RecallThreshold);
        if (!recallCue && recalled.empty())
            return "";
        if (!recalled.empty())
            history = recalled;
    }

    std::string pastRecall = BuildArchiveRecall(botGuid, playerGuid, query);

    if (profile.empty() && history.empty() && pastRecall.empty())
        return "";

    if (profile.empty())
        profile = "-";
    if (history.empty())
        history = "-";

    return SafeFormat(
        kPromptTemplate,
        fmt::arg("player_name", playerName),
        fmt::arg("profile", profile),
        fmt::arg("history", history),
        fmt::arg("past_recall", pastRecall));
}

uint32_t GetPendingTurnCount(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return 0;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return 0;
    uint32_t wm = GetWatermark(botGuid, playerGuid);
    return playerIt->second.size() > wm ? static_cast<uint32_t>(playerIt->second.size() - wm) : 0;
}

void ResetBotMemory(uint64_t botGuid, uint64_t playerGuid)
{
    DrainMemoryWorkersAndQueue();
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (botGuid == 0 && playerGuid == 0)
        {
            g_SemanticMemory.clear();
            g_CompactedTurnCount.clear();
            g_TurnTimestamps.clear();
            g_LastNudgeTime.clear();
            {
                std::lock_guard<std::mutex> cooldownLock(g_CompactionCooldownMutex);
                g_CompactionCooldownUntil.clear();
            }
            g_ArchivePending.clear();
            g_BotConversationHistory.clear();
        }
        else if (playerGuid == 0)
        {
            g_SemanticMemory.erase(botGuid);
            g_CompactedTurnCount.erase(botGuid);
            g_TurnTimestamps.erase(botGuid);
            g_LastNudgeTime.erase(botGuid);
            {
                std::lock_guard<std::mutex> cooldownLock(g_CompactionCooldownMutex);
                g_CompactionCooldownUntil.erase(botGuid);
            }
            g_ArchivePending.erase(botGuid);
            g_BotConversationHistory.erase(botGuid);
        }
        else
        {
            if (g_SemanticMemory.count(botGuid))
                g_SemanticMemory[botGuid].erase(playerGuid);
            if (g_CompactedTurnCount.count(botGuid))
                g_CompactedTurnCount[botGuid].erase(playerGuid);
            if (g_TurnTimestamps.count(botGuid))
                g_TurnTimestamps[botGuid].erase(playerGuid);
            if (g_LastNudgeTime.count(botGuid))
                g_LastNudgeTime[botGuid].erase(playerGuid);
            {
                std::lock_guard<std::mutex> cooldownLock(g_CompactionCooldownMutex);
                if (g_CompactionCooldownUntil.count(botGuid))
                    g_CompactionCooldownUntil[botGuid].erase(playerGuid);
            }
            if (g_ArchivePending.count(botGuid))
                g_ArchivePending[botGuid].erase(playerGuid);
            if (g_BotConversationHistory.count(botGuid))
                g_BotConversationHistory[botGuid].erase(playerGuid);
        }
    }

    if (botGuid == 0 && playerGuid == 0)
    {
        CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
        trans->Append("DELETE FROM mod_ollama_chat_memory_semantic");
        trans->Append("DELETE FROM mod_ollama_chat_memory_episodic");
        trans->Append("DELETE FROM mod_ollama_chat_memory_archive");
        trans->Append("DELETE FROM mod_ollama_chat_history");
        MemoryCommit(trans, "reset all");
    }
    else if (playerGuid == 0)
    {
        CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
        trans->Append(SafeFormat("DELETE FROM mod_ollama_chat_memory_semantic WHERE bot_guid = {}", botGuid));
        trans->Append(SafeFormat("DELETE FROM mod_ollama_chat_memory_episodic WHERE bot_guid = {}", botGuid));
        trans->Append(SafeFormat("DELETE FROM mod_ollama_chat_memory_archive WHERE bot_guid = {}", botGuid));
        trans->Append(SafeFormat("DELETE FROM mod_ollama_chat_history WHERE bot_guid = {}", botGuid));
        MemoryCommit(trans, "reset bot");
    }
    else
    {
        CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memory_semantic WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid));
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memory_episodic WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid));
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_memory_archive WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid));
        trans->Append(SafeFormat(
            "DELETE FROM mod_ollama_chat_history WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid));
        MemoryCommit(trans, "reset pair");
    }
}

std::vector<std::pair<uint64_t, uint64_t>> CollectMemoryPairs(uint64_t botFilter, uint64_t playerFilter)
{
    std::set<std::pair<uint64_t, uint64_t>> keys;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        for (auto const& [botGuid, playerMap] : g_SemanticMemory)
            for (auto const& [playerGuid, _] : playerMap)
                keys.emplace(botGuid, playerGuid);
        for (auto const& [botGuid, playerMap] : g_BotConversationHistory)
            for (auto const& [playerGuid, _] : playerMap)
                keys.emplace(botGuid, playerGuid);
    }

    if (QueryResult result = CharacterDatabase.Query(
            "SELECT bot_guid, player_guid FROM mod_ollama_chat_memory_semantic "
            "UNION SELECT bot_guid, player_guid FROM mod_ollama_chat_memory_episodic"))
    {
        do
        {
            keys.emplace((*result)[0].Get<uint64_t>(), (*result)[1].Get<uint64_t>());
        } while (result->NextRow());
    }

    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    pairs.reserve(keys.size());
    for (auto const& [botGuid, playerGuid] : keys)
    {
        if (botFilter && botGuid != botFilter)
            continue;
        if (playerFilter && playerGuid != playerFilter)
            continue;
        pairs.emplace_back(botGuid, playerGuid);
    }
    return pairs;
}

std::string GetStoredPlayerName(uint64_t botGuid, uint64_t playerGuid)
{
    if (Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid)))
        return player->GetName();
    if (QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT player_name FROM mod_ollama_chat_memory_semantic "
            "WHERE bot_guid = {} AND player_guid = {} AND player_name != '' LIMIT 1",
            botGuid, playerGuid)))
    {
        return (*result)[0].Get<std::string>();
    }
    if (QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT player_name FROM mod_ollama_chat_memory_episodic "
            "WHERE bot_guid = {} AND player_guid = {} AND player_name != '' LIMIT 1",
            botGuid, playerGuid)))
    {
        return (*result)[0].Get<std::string>();
    }
    return "";
}

std::string GetMemoryDebugInfo(uint64_t botGuid, uint64_t playerGuid)
{
    std::string mem;
    uint32_t wm = 0;
    uint32_t pending = 0;
    uint32_t archivePending = 0;
    MemorySections sections;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        mem = GetSemanticText(botGuid, playerGuid);
        sections = ParseMemorySections(mem);
        wm = GetWatermark(botGuid, playerGuid);
        auto histIt = g_BotConversationHistory.find(botGuid);
        if (histIt != g_BotConversationHistory.end())
        {
            auto pit = histIt->second.find(playerGuid);
            if (pit != histIt->second.end() && pit->second.size() > wm)
                pending = static_cast<uint32_t>(pit->second.size() - wm);
        }
        auto ap = g_ArchivePending.find(botGuid);
        if (ap != g_ArchivePending.end())
        {
            auto pit = ap->second.find(playerGuid);
            if (pit != ap->second.end())
                archivePending = static_cast<uint32_t>(pit->second.size());
        }
    }

    uint32_t archiveRows = 0;
    std::string lastCompacted = "never";
    if (QueryResult ar = CharacterDatabase.Query(SafeFormat(
            "SELECT COUNT(*) FROM mod_ollama_chat_memory_archive WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid)))
    {
        archiveRows = (*ar)[0].Get<uint32_t>();
    }
    if (QueryResult lc = CharacterDatabase.Query(SafeFormat(
            "SELECT last_compacted_at FROM mod_ollama_chat_memory_semantic "
            "WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid)))
    {
        if (!(*lc)[0].IsNull())
            lastCompacted = (*lc)[0].Get<std::string>();
    }

    return fmt::format(
        "watermark={} pending={} archive_pending={} archive_rows={} last_compacted={}\n"
        "[PROFILE]\n{}\n[HISTORY]\n{}",
        wm, pending, archivePending, archiveRows, lastCompacted,
        sections.profile.empty() ? "(empty)" : sections.profile,
        sections.history.empty() ? "(empty)" : sections.history);
}
