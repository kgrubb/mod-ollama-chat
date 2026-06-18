#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include "DatabaseEnv.h"
#include "ObjectAccessor.h"
#include "Player.h"
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
#include <cmath>
#include <deque>

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

// Fire-and-forget commit. Never blocks, so it is safe to call on the world thread.
static void MemoryCommitAsync(CharacterDatabaseTransaction const& trans)
{
    if (!trans || trans->GetSize() == 0)
        return;

    CharacterDatabase.CommitTransaction(trans);
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

static void TrimArchivePair(uint64_t botGuid, uint64_t playerGuid);
static void InsertArchiveTurnsToDB(uint64_t botGuid, uint64_t playerGuid,
    std::vector<std::pair<std::string, std::string>> const& turns);
static void LoadPairFromDB(uint64_t botGuid, uint64_t playerGuid);

struct ArchiveTurn
{
    std::string playerMsg;
    std::string botReply;
    time_t at = 0;
};

struct PairMemoryState
{
    std::string facts;
    std::string notes;
    std::string playerName;
    std::deque<ArchiveTurn> archiveRing;
    time_t lastPlayerAt = 0;
    time_t cacheLastUsed = 0;
    time_t lastMaintenanceAt = 0;
};

std::unordered_map<uint64_t, std::unordered_map<uint64_t, PairMemoryState>> g_PairMemory;
std::deque<std::pair<uint64_t, uint64_t>> g_PairLoadQueue;
std::unordered_set<uint64_t> g_PairLoadPending;

std::unordered_map<uint64_t, std::unordered_map<uint64_t, time_t>> g_MaintenanceFailCooldownUntil;
std::mutex g_MaintenanceFailCooldownMutex;

constexpr char kFactsMarker[] = "[FACTS]";
constexpr char kNotesMarker[] = "[NOTES]";
constexpr char kProfileMarker[] = "[PROFILE]";
constexpr char kHistoryMarker[] = "[HISTORY]";

constexpr char kPromptTemplate[] =
    "Memory of {name}:\n"
    "Facts: {facts}\n"
    "{notes_line}"
    "{past_section}";

uint64_t MemoryPairKey(uint64_t botGuid, uint64_t playerGuid)
{
    return botGuid ^ (playerGuid * 2654435761ULL);
}

uint64_t MemoryJobKey(MemoryJobType type, uint64_t botGuid, uint64_t playerGuid)
{
    return MemoryPairKey(botGuid, playerGuid) ^ (static_cast<uint64_t>(type) << 32);
}

static uint32_t GetWatermark(uint64_t botGuid, uint64_t playerGuid);
static void SetWatermark(uint64_t botGuid, uint64_t playerGuid, uint32_t count);

static bool IsMaintenanceFailCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_MaintenanceFailCooldownMutex);
    auto botIt = g_MaintenanceFailCooldownUntil.find(botGuid);
    if (botIt == g_MaintenanceFailCooldownUntil.end())
        return false;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return false;
    return time(nullptr) < playerIt->second;
}

static void SetMaintenanceFailCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_MaintenanceFailCooldownMutex);
    g_MaintenanceFailCooldownUntil[botGuid][playerGuid] =
        time(nullptr) + static_cast<time_t>(OllamaMemory::CompactionFailCooldownSeconds);
}

static void ClearMaintenanceFailCooldown(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_MaintenanceFailCooldownMutex);
    auto botIt = g_MaintenanceFailCooldownUntil.find(botGuid);
    if (botIt != g_MaintenanceFailCooldownUntil.end())
        botIt->second.erase(playerGuid);
}

static std::string TruncateMemory(std::string const& text, uint32_t maxChars)
{
    if (text.size() <= maxChars)
        return text;
    return text.substr(0, maxChars);
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

static std::string ExtractSection(std::string const& blob, char const* marker, char const* nextMarker)
{
    size_t pos = blob.find(marker);
    if (pos == std::string::npos)
        return "";
    size_t start = pos + strlen(marker);
    while (start < blob.size() && (blob[start] == '\n' || blob[start] == '\r' || blob[start] == ' '))
        ++start;
    size_t end = blob.size();
    if (nextMarker)
    {
        size_t nPos = blob.find(nextMarker, start);
        if (nPos != std::string::npos)
            end = nPos;
    }
    std::string sec = blob.substr(start, end - start);
    while (!sec.empty() && (sec.back() == '\n' || sec.back() == '\r'))
        sec.pop_back();
    size_t trim = sec.find_first_not_of(" \t\r\n");
    if (trim != std::string::npos)
        sec = sec.substr(trim);
    return sec;
}

static std::string MigrateLegacyMemoryText(std::string const& memoryText)
{
    if (memoryText.empty())
        return "";

    std::string history = ExtractSection(memoryText, kHistoryMarker, kProfileMarker);
    if (history.empty())
        history = ExtractSection(memoryText, kHistoryMarker, nullptr);
    std::string profile = ExtractSection(memoryText, kProfileMarker, kHistoryMarker);
    if (profile.empty())
        profile = ExtractSection(memoryText, kProfileMarker, nullptr);

    std::string notes;
    if (!history.empty())
        notes = TruncateMemory(history, 200);
    if (!profile.empty())
    {
        if (!notes.empty())
            notes += "\n";
        notes += profile;
        notes = TruncateMemory(notes, 256);
    }
    return notes;
}

static bool ParseMaintenanceResponse(std::string const& raw, std::string& facts, std::string& notes)
{
    std::string s = TrimMemoryResponse(raw);
    if (s.empty())
        return false;
    size_t fPos = s.find(kFactsMarker);
    size_t nPos = s.find(kNotesMarker);
    if (fPos == std::string::npos && nPos == std::string::npos)
        return false;
    facts = ExtractSection(s, kFactsMarker, kNotesMarker);
    notes = ExtractSection(s, kNotesMarker, nullptr);
    return true;
}

static void AppendDialogueTurn(std::ostringstream& out, std::string const& playerName,
    std::string const& botName, ConversationTurn const& turn, bool tagUnverified)
{
    out << playerName << ": " << turn.playerMessage << "\n";
    out << botName << ": " << turn.botReply;
    if (tagUnverified && !turn.verified)
        out << " (unverified)";
    out << "\n";
}

static std::vector<std::string> Tokenize(std::string const& text)
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

static std::string ArchiveDoc(ArchiveTurn const& turn)
{
    return "Player: " + turn.playerMsg + "\nBot: " + turn.botReply;
}

static std::string RetrieveBM25(std::string const& query, std::vector<std::string> const& docs, uint32_t maxItems)
{
    if (docs.empty() || maxItems == 0)
        return "";

    auto qTok = Tokenize(query);
    if (qTok.empty())
        return "";

    std::vector<std::unordered_map<std::string, float>> tfs(docs.size());
    std::vector<uint32_t> lens(docs.size());
    std::unordered_map<std::string, uint32_t> df;
    for (size_t i = 0; i < docs.size(); ++i)
    {
        auto toks = Tokenize(docs[i]);
        lens[i] = static_cast<uint32_t>(toks.size());
        for (auto const& t : toks)
        {
            tfs[i][t] += 1.f;
            ++df[t];
        }
    }

    float avgLen = 0.f;
    for (auto l : lens)
        avgLen += static_cast<float>(l);
    avgLen /= static_cast<float>(docs.size() > 0 ? docs.size() : 1);

    constexpr float k1 = 1.2f;
    constexpr float b = 0.75f;
    std::vector<std::pair<size_t, float>> scored;
    scored.reserve(docs.size());
    for (size_t i = 0; i < docs.size(); ++i)
    {
        float score = 0.f;
        for (auto const& qt : qTok)
        {
            auto it = tfs[i].find(qt);
            if (it == tfs[i].end())
                continue;
            float tf = it->second;
            uint32_t n = df[qt];
            float idf = std::log((docs.size() - n + 0.5f) / (n + 0.5f) + 1.f);
            float denom = tf + k1 * (1.f - b + b * static_cast<float>(lens[i]) / avgLen);
            score += idf * (tf * (k1 + 1.f)) / denom;
        }
        if (score > 0.f)
            scored.emplace_back(i, score);
    }

    if (scored.empty())
        return "";

    if (scored.size() > maxItems)
    {
        std::partial_sort(scored.begin(), scored.begin() + maxItems, scored.end(),
            [](auto const& a, auto const& b) { return a.second > b.second; });
        scored.resize(maxItems);
    }
    else
    {
        std::sort(scored.begin(), scored.end(), [](auto const& a, auto const& b) { return a.second > b.second; });
    }

    std::ostringstream out;
    for (auto const& [idx, _] : scored)
        out << docs[idx] << "\n";
    std::string r = out.str();
    while (!r.empty() && (r.back() == '\n' || r.back() == '\r'))
        r.pop_back();
    return r;
}

static bool NeedsEpisodicRecall(ChatChannelSourceLocal channel, time_t lastPlayerAt,
    std::deque<ArchiveTurn> const& archiveRing, uint32_t pending, std::string const& message)
{
    if (channel == SRC_GENERAL_LOCAL)
        return false;

    if (lastPlayerAt && difftime(time(nullptr), lastPlayerAt) >=
            static_cast<double>(OllamaMemory::SessionResumeMinutes) * 60.0)
        return true;
    if (pending <= 2 && !archiveRing.empty())
        return true;
    return message.find('?') != std::string::npos;
}

static bool IsSessionResume(time_t lastPlayerAt)
{
    return lastPlayerAt && difftime(time(nullptr), lastPlayerAt) >=
        static_cast<double>(OllamaMemory::SessionResumeMinutes) * 60.0;
}

static std::string BuildPastRecall(time_t lastPlayerAt, std::deque<ArchiveTurn> const& archiveRing,
    std::string const& message)
{
    if (archiveRing.empty())
        return "";

    if (IsSessionResume(lastPlayerAt))
    {
        std::ostringstream out;
        size_t n = std::min<size_t>(3, archiveRing.size());
        for (size_t i = archiveRing.size() - n; i < archiveRing.size(); ++i)
            out << ArchiveDoc(archiveRing[i]) << "\n";
        std::string r = out.str();
        while (!r.empty() && (r.back() == '\n' || r.back() == '\r'))
            r.pop_back();
        return r;
    }

    std::vector<std::string> docs;
    docs.reserve(archiveRing.size());
    for (auto const& turn : archiveRing)
        docs.push_back(ArchiveDoc(turn));
    return RetrieveBM25(message, docs, OllamaMemory::RecallMaxItems);
}

static PairMemoryState* GetPairUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_PairMemory.find(botGuid);
    if (botIt == g_PairMemory.end())
        return nullptr;
    auto pit = botIt->second.find(playerGuid);
    return pit == botIt->second.end() ? nullptr : &pit->second;
}

static PairMemoryState& TouchPairUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto& pair = g_PairMemory[botGuid][playerGuid];
    pair.cacheLastUsed = time(nullptr);
    return pair;
}

static bool PairDequeEmptyUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return true;
    auto pit = botIt->second.find(playerGuid);
    return pit == botIt->second.end() || pit->second.empty();
}

static bool IsPlayerOffline(uint64_t playerGuid)
{
    return !ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
}

static bool PairIdle(PairMemoryState const& state, time_t now)
{
    if (!state.cacheLastUsed)
        return false;
    return difftime(now, state.cacheLastUsed) >= static_cast<double>(OllamaMemory::PairCacheIdleSec);
}

static void EvictPairCacheUnlocked()
{
    time_t now = time(nullptr);
    if (g_PairMemory.size() <= OllamaMemory::PairCacheMax)
    {
        for (auto botIt = g_PairMemory.begin(); botIt != g_PairMemory.end();)
        {
            for (auto pit = botIt->second.begin(); pit != botIt->second.end();)
            {
                uint64_t playerGuid = pit->first;
                if (IsPlayerOffline(playerGuid) && PairDequeEmptyUnlocked(botIt->first, playerGuid) &&
                    PairIdle(pit->second, now))
                {
                    pit = botIt->second.erase(pit);
                }
                else
                    ++pit;
            }
            if (botIt->second.empty())
                botIt = g_PairMemory.erase(botIt);
            else
                ++botIt;
        }
        return;
    }

    struct Candidate
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        time_t lastUsed;
    };
    std::vector<Candidate> candidates;
    for (auto const& [botGuid, playerMap] : g_PairMemory)
    {
        for (auto const& [playerGuid, state] : playerMap)
        {
            if (IsPlayerOffline(playerGuid) && PairDequeEmptyUnlocked(botGuid, playerGuid))
                candidates.push_back({ botGuid, playerGuid, state.cacheLastUsed });
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](Candidate const& a, Candidate const& b) { return a.lastUsed < b.lastUsed; });

    size_t toEvict = g_PairMemory.size() - OllamaMemory::PairCacheMax + 1;
    for (size_t i = 0; i < candidates.size() && toEvict > 0; ++i, --toEvict)
        g_PairMemory[candidates[i].botGuid].erase(candidates[i].playerGuid);
}

static void PushArchiveTurnUnlocked(PairMemoryState& pair, std::string const& playerMsg,
    std::string const& botReply, time_t at)
{
    pair.archiveRing.push_back({ playerMsg, botReply, at });
    while (pair.archiveRing.size() > OllamaMemory::ArchiveRingCap)
        pair.archiveRing.pop_front();
}

static void EnqueuePairLoad(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    uint64_t key = MemoryPairKey(botGuid, playerGuid);
    if (g_PairLoadPending.count(key))
        return;
    g_PairLoadPending.insert(key);
    g_PairLoadQueue.push_back({ botGuid, playerGuid });
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

static uint32_t PendingCountUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return 0;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return 0;
    uint32_t wm = GetWatermark(botGuid, playerGuid);
    return playerIt->second.size() > wm ? static_cast<uint32_t>(playerIt->second.size() - wm) : 0;
}

static bool IsMaintenanceOnCooldown(PairMemoryState const& pair)
{
    if (!pair.lastMaintenanceAt)
        return false;
    return difftime(time(nullptr), pair.lastMaintenanceAt) <
        static_cast<double>(OllamaMemory::MaintenanceCooldownSec);
}

static bool ShouldRunMaintenanceUnlocked(uint64_t botGuid, uint64_t playerGuid)
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
            if (difftime(time(nullptr), oldest) >=
                    static_cast<double>(OllamaMemory::CompactionMaxAgeMinutes) * 60.0)
                return true;
        }
    }
    return false;
}

static bool AllPendingTrivialUnlocked(uint64_t botGuid, uint64_t playerGuid)
{
    auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return true;
    auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return true;

    uint32_t wm = GetWatermark(botGuid, playerGuid);
    for (size_t i = wm; i < playerIt->second.size(); ++i)
    {
        if (!IsTrivialTurn(playerIt->second[i].playerMessage))
            return false;
    }
    return true;
}

static bool DequeueMemoryJob(MemoryJob& job)
{
    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    if (g_MemoryMaintenanceQueue.empty())
        return false;
    job = g_MemoryMaintenanceQueue.front();
    g_MemoryMaintenanceQueue.pop_front();
    return true;
}

static bool RunMemoryMaintenance(uint64_t botGuid, uint64_t playerGuid)
{
    std::vector<ConversationTurn> pendingTurns;
    std::string existingFacts;
    std::string existingNotes;
    std::string storedName;
    uint32_t pendingCount = 0;
    uint32_t snapshotWm = 0;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return false;
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt == botIt->second.end())
            return false;

        snapshotWm = GetWatermark(botGuid, playerGuid);
        size_t dequeSize = playerIt->second.size();
        if (dequeSize <= snapshotWm)
            return false;

        pendingCount = static_cast<uint32_t>(dequeSize - snapshotWm);
        pendingTurns.assign(playerIt->second.begin() + static_cast<std::ptrdiff_t>(snapshotWm), playerIt->second.end());

        if (PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid))
        {
            existingFacts = pair->facts;
            existingNotes = pair->notes;
            storedName = pair->playerName;
        }
    }

    if (g_queryManager.ShouldDeferMemoryWork())
    {
        std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
        g_MemoryMaintenanceQueue.push_front({ MemoryJobType::Maintenance, botGuid, playerGuid });
        return true;
    }

    Player* bot = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
    Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
    std::string playerName = player ? player->GetName() : (storedName.empty() ? "Player" : storedName);
    std::string botName = bot ? bot->GetName() : "Bot";

    std::ostringstream turnsBlock;
    for (ConversationTurn const& turn : pendingTurns)
        AppendDialogueTurn(turnsBlock, playerName, botName, turn, !turn.verified);

    ScenarioInput input;
    input.maintenanceFacts = existingFacts;
    input.maintenanceNotes = existingNotes;
    input.maintenanceTurns = turnsBlock.str();

    PromptBundle bundle = OllamaPromptComposer::Build(PromptScenario::MemoryMaintenance, {}, input);
    bundle.maxTokens = OllamaMemory::MemoryQueryMaxTokens;
    auto future = SubmitQuery(bundle);
    std::string response = future.valid() ? future.get() : "";

    std::string newFacts = existingFacts;
    std::string newNotes = existingNotes;
    bool parsed = false;
    if (!response.empty())
    {
        std::string llmFacts;
        std::string llmNotes;
        parsed = ParseMaintenanceResponse(response, llmFacts, llmNotes);
        if (parsed)
        {
            if (!llmFacts.empty())
                newFacts = llmFacts;
            if (!llmNotes.empty())
                newNotes = llmNotes;
        }
    }

    bool const failed = response.empty() || !parsed;
    newFacts = TruncateMemory(newFacts, OllamaMemory::FactsMaxChars);
    newNotes = TruncateMemory(newNotes, OllamaMemory::NotesMaxChars);

    time_t now = time(nullptr);
    std::vector<std::pair<std::string, std::string>> archiveInserts;
    archiveInserts.reserve(pendingTurns.size());
    for (auto const& turn : pendingTurns)
        archiveInserts.emplace_back(turn.playerMessage, turn.botReply);

    uint32_t newWm = 0;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return false;
        auto playerIt = botIt->second.find(playerGuid);
        if (playerIt == botIt->second.end() || playerIt->second.size() < snapshotWm)
            return false;

        PairMemoryState& pair = TouchPairUnlocked(botGuid, playerGuid);
        if (failed)
        {
            newFacts = existingFacts;
            newNotes = existingNotes;
        }
        else
        {
            pair.facts = newFacts;
            pair.notes = newNotes;
            pair.lastMaintenanceAt = now;
            ClearMaintenanceFailCooldown(botGuid, playerGuid);
        }

        pair.playerName = playerName;
        for (auto const& turn : pendingTurns)
            PushArchiveTurnUnlocked(pair, turn.playerMessage, turn.botReply, now);

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

        EvictPairCacheUnlocked();
    }

    if (failed)
    {
        SetMaintenanceFailCooldown(botGuid, playerGuid);
        LOG_WARN("server.loading", "[OllamaChat] Memory maintenance failed bot {} player {}: {}",
            botGuid, playerGuid, response.empty() ? "empty LLM response" : "parse fail");
    }

    std::string escFacts = newFacts;
    std::string escNotes = newNotes;
    std::string escName = playerName;
    CharacterDatabase.EscapeString(escFacts);
    CharacterDatabase.EscapeString(escNotes);
    CharacterDatabase.EscapeString(escName);

    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    for (auto const& turn : archiveInserts)
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

    std::string lastAtSql = "NULL";
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid))
        {
            if (pair->lastPlayerAt)
                lastAtSql = SafeFormat("FROM_UNIXTIME({})", static_cast<uint64_t>(pair->lastPlayerAt));
        }
    }

    trans->Append(SafeFormat(
        "DELETE FROM mod_ollama_chat_memory_episodic WHERE bot_guid = {} AND player_guid = {}",
        botGuid, playerGuid));

    trans->Append(SafeFormat(
        "REPLACE INTO mod_ollama_chat_memory_semantic (bot_guid, player_guid, player_name, facts_text, notes_text, "
        "last_player_at, compacted_turn_count, last_compacted_at) VALUES "
        "({}, {}, '{}', '{}', '{}', {}, {}, NOW())",
        botGuid, playerGuid, escName, escFacts, escNotes, lastAtSql, newWm));

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt != g_BotConversationHistory.end())
        {
            auto playerIt = botIt->second.find(playerGuid);
            if (playerIt != botIt->second.end())
            {
                uint32_t wm = GetWatermark(botGuid, playerGuid);
                for (size_t i = wm; i < playerIt->second.size(); ++i)
                {
                    std::string escCtx = playerIt->second[i].playerMessage;
                    std::string escReply = playerIt->second[i].botReply;
                    CharacterDatabase.EscapeString(escCtx);
                    CharacterDatabase.EscapeString(escReply);
                    trans->Append(SafeFormat(
                        "INSERT INTO mod_ollama_chat_memory_episodic (bot_guid, player_guid, player_name, context, bot_reply) "
                        "VALUES ({}, {}, '{}', '{}', '{}')",
                        botGuid, playerGuid, escName, escCtx, escReply));
                }
            }
        }
    }

    MemoryCommit(trans, "maintenance");
    TrimArchivePair(botGuid, playerGuid);

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] Memory maintenance bot {} player {} ({} turns, {})",
            botGuid, playerGuid, pendingCount, failed ? "failed" : "ok");
    return false;
}

static void DispatchMemoryJob(MemoryJob const& job)
{
    uint64_t jobKey = MemoryJobKey(job.type, job.botGuid, job.playerGuid);
    ++g_MemoryMaintenanceInFlight;
    std::thread([job, jobKey]() {
        bool const deferred = RunMemoryMaintenance(job.botGuid, job.playerGuid);
        --g_MemoryMaintenanceInFlight;
        if (deferred)
            return;
        std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
        g_MemoryMaintenancePending.erase(jobKey);
    }).detach();
}

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

static std::string ArchiveTrimSql(uint64_t botGuid, uint64_t playerGuid)
{
    return SafeFormat(
        "DELETE FROM mod_ollama_chat_memory_archive "
        "WHERE bot_guid = {} AND player_guid = {} AND id NOT IN ("
        "  SELECT id FROM ("
        "    SELECT id FROM mod_ollama_chat_memory_archive "
        "    WHERE bot_guid = {} AND player_guid = {} "
        "    ORDER BY created_at DESC LIMIT {}"
        "  ) t"
        ")",
        botGuid, playerGuid, botGuid, playerGuid, OllamaMemory::ArchiveMaxRowsPerPair);
}

static void TrimArchivePair(uint64_t botGuid, uint64_t playerGuid)
{
    MemoryExecute(ArchiveTrimSql(botGuid, playerGuid), "archive trim");
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

static void LoadPairFromDB(uint64_t botGuid, uint64_t playerGuid)
{
    PairMemoryState loaded;
    bool hasRow = false;
    uint32_t wm = 0;

    if (QueryResult result = CharacterDatabase.Query(SafeFormat(
            "SELECT player_name, memory_text, facts_text, notes_text, "
            "UNIX_TIMESTAMP(last_player_at), compacted_turn_count "
            "FROM mod_ollama_chat_memory_semantic WHERE bot_guid = {} AND player_guid = {}",
            botGuid, playerGuid)))
    {
        loaded.playerName = (*result)[0].Get<std::string>();
        std::string memoryText = (*result)[1].Get<std::string>();
        loaded.facts = (*result)[2].Get<std::string>();
        loaded.notes = (*result)[3].Get<std::string>();
        if (!(*result)[4].IsNull())
            loaded.lastPlayerAt = static_cast<time_t>((*result)[4].Get<uint64_t>());
        wm = (*result)[5].Get<uint32_t>();
        hasRow = true;

        if (loaded.facts.empty() && !memoryText.empty())
            loaded.notes = MigrateLegacyMemoryText(memoryText);
    }

    if (QueryResult ar = CharacterDatabase.Query(SafeFormat(
            "SELECT context, bot_reply, UNIX_TIMESTAMP(created_at) "
            "FROM mod_ollama_chat_memory_archive WHERE bot_guid = {} AND player_guid = {} "
            "ORDER BY created_at DESC LIMIT {}",
            botGuid, playerGuid, OllamaMemory::ArchiveRecallWindow)))
    {
        std::vector<ArchiveTurn> rows;
        do
        {
            ArchiveTurn turn;
            turn.playerMsg = (*ar)[0].Get<std::string>();
            turn.botReply = (*ar)[1].Get<std::string>();
            turn.at = (*ar)[2].IsNull() ? time(nullptr) : static_cast<time_t>((*ar)[2].Get<uint64_t>());
            rows.push_back(std::move(turn));
        } while (ar->NextRow());

        std::reverse(rows.begin(), rows.end());
        for (auto const& turn : rows)
            loaded.archiveRing.push_back(turn);
        while (loaded.archiveRing.size() > OllamaMemory::ArchiveRingCap)
            loaded.archiveRing.pop_front();
    }

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        PairMemoryState& pair = TouchPairUnlocked(botGuid, playerGuid);
        if (hasRow)
        {
            g_CompactedTurnCount[botGuid][playerGuid] = wm;
            if (pair.facts.empty())
                pair.facts = loaded.facts;
            if (pair.notes.empty())
                pair.notes = loaded.notes;
            if (pair.playerName.empty())
                pair.playerName = loaded.playerName;
            if (!pair.lastPlayerAt)
                pair.lastPlayerAt = loaded.lastPlayerAt;
        }
        if (pair.archiveRing.empty() && !loaded.archiveRing.empty())
            pair.archiveRing = std::move(loaded.archiveRing);
        EvictPairCacheUnlocked();
    }
}

static void ProcessPairLoadTick()
{
    std::pair<uint64_t, uint64_t> pairKey;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (g_PairLoadQueue.empty())
            return;
        pairKey = g_PairLoadQueue.front();
        g_PairLoadQueue.pop_front();
        g_PairLoadPending.erase(MemoryPairKey(pairKey.first, pairKey.second));
    }
    LoadPairFromDB(pairKey.first, pairKey.second);
}

static void WaitForMemoryWorkersBlocking()
{
    uint32_t i = 0;
    for (; i < 1200 && g_MemoryMaintenanceInFlight.load() > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (i >= 1200 && g_MemoryMaintenanceInFlight.load() > 0)
        LOG_WARN("server.loading", "[OllamaChat] Memory worker wait timed out ({} in flight)",
            g_MemoryMaintenanceInFlight.load());
}

static void DrainMemoryWorkersAndQueue()
{
    WaitForMemoryWorkersBlocking();
    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    g_MemoryMaintenanceQueue.clear();
    g_MemoryMaintenancePending.clear();
}

static void LoadBotMemoryFromDB();

void InitializeBotMemory()
{
    if (!g_EnableMemory)
        return;
    DrainMemoryWorkersAndQueue();
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        g_PairMemory.clear();
        g_PairLoadQueue.clear();
        g_PairLoadPending.clear();
        {
            std::lock_guard<std::mutex> cooldownLock(g_MaintenanceFailCooldownMutex);
            g_MaintenanceFailCooldownUntil.clear();
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
        g_CompactedTurnCount.clear();
        g_TurnTimestamps.clear();
        g_BotConversationHistory.clear();
        g_PairMemory.clear();

        if (QueryResult result = CharacterDatabase.Query(
                "SELECT bot_guid, player_guid, player_name, memory_text, facts_text, notes_text, "
                "UNIX_TIMESTAMP(last_player_at), compacted_turn_count "
                "FROM mod_ollama_chat_memory_semantic"))
        {
            do
            {
                uint64_t botGuid = (*result)[0].Get<uint64_t>();
                uint64_t playerGuid = (*result)[1].Get<uint64_t>();
                std::string memoryText = (*result)[3].Get<std::string>();
                std::string facts = (*result)[4].Get<std::string>();
                std::string notes = (*result)[5].Get<std::string>();

                g_CompactedTurnCount[botGuid][playerGuid] = (*result)[7].Get<uint32_t>();

                PairMemoryState& pair = g_PairMemory[botGuid][playerGuid];
                pair.facts = facts;
                pair.notes = notes;
                if (facts.empty() && !memoryText.empty())
                    pair.notes = MigrateLegacyMemoryText(memoryText);
                pair.playerName = (*result)[2].Get<std::string>();
                if (!(*result)[6].IsNull())
                    pair.lastPlayerAt = static_cast<time_t>((*result)[6].Get<uint64_t>());
                pair.cacheLastUsed = time(nullptr);
            } while (result->NextRow());
        }

        if (QueryResult arch = CharacterDatabase.Query(
                "SELECT bot_guid, player_guid, context, bot_reply, UNIX_TIMESTAMP(created_at) "
                "FROM mod_ollama_chat_memory_archive ORDER BY created_at ASC"))
        {
            do
            {
                uint64_t botGuid = (*arch)[0].Get<uint64_t>();
                uint64_t playerGuid = (*arch)[1].Get<uint64_t>();
                ArchiveTurn turn;
                turn.playerMsg = (*arch)[2].Get<std::string>();
                turn.botReply = (*arch)[3].Get<std::string>();
                turn.at = (*arch)[4].IsNull() ? time(nullptr) : static_cast<time_t>((*arch)[4].Get<uint64_t>());
                auto& pair = g_PairMemory[botGuid][playerGuid];
                pair.archiveRing.push_back(std::move(turn));
                while (pair.archiveRing.size() > OllamaMemory::ArchiveRecallWindow)
                    pair.archiveRing.pop_front();
            } while (arch->NextRow());

            for (auto& [botGuid, playerMap] : g_PairMemory)
            {
                for (auto& [playerGuid, pair] : playerMap)
                {
                    while (pair.archiveRing.size() > OllamaMemory::ArchiveRingCap)
                        pair.archiveRing.pop_front();
                }
            }
        }

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
                if (!hist.empty() && !g_CompactedTurnCount.count(botGuid))
                    g_CompactedTurnCount[botGuid][playerGuid] = 0;
                else if (!hist.empty() && !g_CompactedTurnCount[botGuid].count(playerGuid))
                    g_CompactedTurnCount[botGuid][playerGuid] = 0;
            }
        }

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
    // Only the shutdown path waits for workers. The periodic save runs on the
    // world thread and must not sleep there.
    if (blocking)
        WaitForMemoryWorkersBlocking();

    struct SemanticSaveRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::string facts;
        std::string notes;
        std::string playerName;
        time_t lastPlayerAt;
        uint32_t watermark;
    };

    struct EpisodicSaveRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::vector<std::pair<std::string, std::string>> pendingTurns;
    };

    std::vector<SemanticSaveRow> semanticRows;
    std::vector<EpisodicSaveRow> episodicRows;
    std::set<std::pair<uint64_t, uint64_t>> trimPairs;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

        for (auto const& [botGuid, playerMap] : g_PairMemory)
        {
            for (auto const& [playerGuid, pair] : playerMap)
            {
                semanticRows.push_back({
                    botGuid,
                    playerGuid,
                    pair.facts,
                    pair.notes,
                    pair.playerName,
                    pair.lastPlayerAt,
                    GetWatermark(botGuid, playerGuid)
                });
                trimPairs.emplace(botGuid, playerGuid);
            }
        }

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
                trimPairs.emplace(botGuid, playerGuid);
            }
        }
    }

    // Batch every pair into one transaction so each cycle does a single commit
    // rather than one blocking commit per pair.
    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();

    for (auto const& row : semanticRows)
    {
        std::string escFacts = row.facts;
        std::string escNotes = row.notes;
        std::string escName = row.playerName.empty() ? GetStoredPlayerName(row.botGuid, row.playerGuid) : row.playerName;
        CharacterDatabase.EscapeString(escFacts);
        CharacterDatabase.EscapeString(escNotes);
        CharacterDatabase.EscapeString(escName);
        std::string lastAtSql = row.lastPlayerAt ? SafeFormat("FROM_UNIXTIME({})", static_cast<uint64_t>(row.lastPlayerAt)) : "NULL";
        trans->Append(SafeFormat(
            "REPLACE INTO mod_ollama_chat_memory_semantic (bot_guid, player_guid, player_name, facts_text, notes_text, "
            "last_player_at, compacted_turn_count, last_compacted_at) VALUES "
            "({}, {}, '{}', '{}', '{}', {}, {}, NOW())",
            row.botGuid, row.playerGuid, escName, escFacts, escNotes, lastAtSql, row.watermark));
    }

    for (auto const& row : episodicRows)
    {
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
    }

    for (auto const& [botGuid, playerGuid] : trimPairs)
        trans->Append(ArchiveTrimSql(botGuid, playerGuid));

    if (blocking)
        MemoryCommit(trans, "save memory");
    else
        MemoryCommitAsync(trans);
    return true;
}

MemoryCompactionEnqueueResult EnqueueMemoryCompaction(uint64_t botGuid, uint64_t playerGuid, bool force)
{
    if (!g_EnableMemory)
        return MemoryCompactionEnqueueResult::NothingPending;

    uint64_t key = MemoryJobKey(MemoryJobType::Maintenance, botGuid, playerGuid);
    uint32_t pending = 0;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        pending = PendingCountUnlocked(botGuid, playerGuid);
        if (force && pending == 0)
        {
            LOG_WARN("server.loading", "[OllamaChat] Force maintenance skipped: no pending turns bot {} player {}",
                botGuid, playerGuid);
            return MemoryCompactionEnqueueResult::NothingPending;
        }
        if (!force && !ShouldRunMaintenanceUnlocked(botGuid, playerGuid))
            return MemoryCompactionEnqueueResult::NothingPending;
    }

    if (!force && IsMaintenanceFailCooldown(botGuid, playerGuid) && pending < OllamaMemory::CompactionThreshold)
        return MemoryCompactionEnqueueResult::NothingPending;

    std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
    if (g_MemoryMaintenancePending.count(key))
        return MemoryCompactionEnqueueResult::AlreadyPending;

    if (g_MemoryMaintenanceQueue.size() >= OllamaMemory::MaxCompactionQueue)
        g_MemoryMaintenanceQueue.pop_back();

    bool online = !IsPlayerOffline(playerGuid);
    MemoryJob job{ MemoryJobType::Maintenance, botGuid, playerGuid };
    if (online)
        g_MemoryMaintenanceQueue.push_front(job);
    else
        g_MemoryMaintenanceQueue.push_back(job);

    g_MemoryMaintenancePending.insert(key);
    return MemoryCompactionEnqueueResult::Enqueued;
}

void MaybeEnqueueMemoryMaintenance(uint64_t botGuid, uint64_t playerGuid)
{
    if (!g_EnableMemory)
        return;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (!ShouldRunMaintenanceUnlocked(botGuid, playerGuid))
            return;
        if (AllPendingTrivialUnlocked(botGuid, playerGuid))
            return;

        uint32_t pending = PendingCountUnlocked(botGuid, playerGuid);
        if (PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid))
        {
            if (IsMaintenanceOnCooldown(*pair) && pending < OllamaMemory::CompactionThreshold)
                return;
        }
    }

    EnqueueMemoryCompaction(botGuid, playerGuid, false);
}

void AppendBotMemoryTurn(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage,
    std::string const& botReply, bool /*isEvent*/, bool verified)
{
    std::vector<std::pair<std::string, std::string>> toFlush;
    time_t now = time(nullptr);

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto& hist = g_BotConversationHistory[botGuid][playerGuid];
        hist.push_back({ playerMessage, botReply, verified });
        g_TurnTimestamps[botGuid][playerGuid].push_back(now);

        PairMemoryState& pair = TouchPairUnlocked(botGuid, playerGuid);
        pair.lastPlayerAt = now;
        if (pair.playerName.empty())
        {
            if (Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid)))
                pair.playerName = player->GetName();
        }

        while (hist.size() > OllamaMemory::DequeCap)
        {
            ConversationTurn evicted = hist.front();
            hist.pop_front();
            PushArchiveTurnUnlocked(pair, evicted.playerMessage, evicted.botReply, now);
            toFlush.emplace_back(evicted.playerMessage, evicted.botReply);

            auto& ts = g_TurnTimestamps[botGuid][playerGuid];
            if (!ts.empty())
                ts.pop_front();
            uint32_t& wm = g_CompactedTurnCount[botGuid][playerGuid];
            if (wm > 0)
                --wm;
        }

        EvictPairCacheUnlocked();
    }

    InsertArchiveTurnsToDB(botGuid, playerGuid, toFlush);
    MaybeEnqueueMemoryMaintenance(botGuid, playerGuid);
}

void ProcessMemoryCompactionTick()
{
    if (!g_EnableMemory)
        return;

    PurgeStaleMemoryRows();
    ProcessPairLoadTick();

    uint32_t toProcess = OllamaMemory::CompactionsPerTick;
    while (toProcess > 0)
    {
        MemoryJob job{};
        if (!DequeueMemoryJob(job))
            break;

        if (OllamaMemory::MaxConcurrentCompactions > 0 &&
            g_MemoryMaintenanceInFlight.load() >= OllamaMemory::MaxConcurrentCompactions)
        {
            std::lock_guard<std::mutex> lock(g_MemoryQueueMutex);
            g_MemoryMaintenanceQueue.push_front(job);
            break;
        }

        DispatchMemoryJob(job);
        --toProcess;
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
                if (ShouldRunMaintenanceUnlocked(botGuid, playerGuid))
                    ageSweep.emplace_back(botGuid, playerGuid);
                if (ageSweep.size() >= 2)
                    break;
            }
            if (ageSweep.size() >= 2)
                break;
        }
    }
    for (auto const& [botGuid, playerGuid] : ageSweep)
        MaybeEnqueueMemoryMaintenance(botGuid, playerGuid);
}

std::string GetMemoryPromptAddition(uint64_t botGuid, uint64_t playerGuid, ChatChannelSourceLocal channel,
    std::string const& message, std::string const& playerName)
{
    if (!g_EnableMemory)
        return "";

    std::string facts;
    std::string notes;
    std::string name;
    std::deque<ArchiveTurn> archiveCopy;
    time_t lastPlayerAt = 0;
    uint32_t pending = 0;
    bool cold = false;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid);
        if (!pair)
        {
            cold = true;
        }
        else
        {
            pair->cacheLastUsed = time(nullptr);
            facts = pair->facts;
            notes = pair->notes;
            name = pair->playerName;
            lastPlayerAt = pair->lastPlayerAt;
            archiveCopy = pair->archiveRing;
            pending = PendingCountUnlocked(botGuid, playerGuid);
        }
    }

    if (cold)
    {
        EnqueuePairLoad(botGuid, playerGuid);
        return "";
    }

    if (name.empty())
        name = playerName;

    if (channel == SRC_GENERAL_LOCAL)
    {
        if (facts.empty())
            return "";
        return SafeFormat(
            kPromptTemplate,
            fmt::arg("name", name),
            fmt::arg("facts", facts),
            fmt::arg("notes_line", ""),
            fmt::arg("past_section", ""));
    }

    std::string past;
    if (NeedsEpisodicRecall(channel, lastPlayerAt, archiveCopy, pending, message))
    {
        past = BuildPastRecall(lastPlayerAt, archiveCopy, message);
        if (!past.empty())
            past = "Past:\n" + past + "\n";
    }

    if (facts.empty() && notes.empty() && past.empty())
        return "";

    std::string notesLine;
    if (!notes.empty())
        notesLine = "Notes: " + notes + "\n";

    return SafeFormat(
        kPromptTemplate,
        fmt::arg("name", name),
        fmt::arg("facts", facts.empty() ? "-" : facts),
        fmt::arg("notes_line", notesLine),
        fmt::arg("past_section", past));
}

uint32_t GetPendingTurnCount(uint64_t botGuid, uint64_t playerGuid)
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    return PendingCountUnlocked(botGuid, playerGuid);
}

void ResetBotMemory(uint64_t botGuid, uint64_t playerGuid)
{
    DrainMemoryWorkersAndQueue();
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (botGuid == 0 && playerGuid == 0)
        {
            g_CompactedTurnCount.clear();
            g_TurnTimestamps.clear();
            g_BotConversationHistory.clear();
            g_PairMemory.clear();
            g_PairLoadQueue.clear();
            g_PairLoadPending.clear();
            {
                std::lock_guard<std::mutex> cooldownLock(g_MaintenanceFailCooldownMutex);
                g_MaintenanceFailCooldownUntil.clear();
            }
        }
        else if (playerGuid == 0)
        {
            g_CompactedTurnCount.erase(botGuid);
            g_TurnTimestamps.erase(botGuid);
            g_BotConversationHistory.erase(botGuid);
            g_PairMemory.erase(botGuid);
            for (auto it = g_PairLoadQueue.begin(); it != g_PairLoadQueue.end();)
            {
                if (it->first == botGuid)
                {
                    g_PairLoadPending.erase(MemoryPairKey(it->first, it->second));
                    it = g_PairLoadQueue.erase(it);
                }
                else
                    ++it;
            }
            {
                std::lock_guard<std::mutex> cooldownLock(g_MaintenanceFailCooldownMutex);
                g_MaintenanceFailCooldownUntil.erase(botGuid);
            }
        }
        else
        {
            if (g_CompactedTurnCount.count(botGuid))
                g_CompactedTurnCount[botGuid].erase(playerGuid);
            if (g_TurnTimestamps.count(botGuid))
                g_TurnTimestamps[botGuid].erase(playerGuid);
            if (g_BotConversationHistory.count(botGuid))
                g_BotConversationHistory[botGuid].erase(playerGuid);
            if (g_PairMemory.count(botGuid))
                g_PairMemory[botGuid].erase(playerGuid);
            g_PairLoadPending.erase(MemoryPairKey(botGuid, playerGuid));
            for (auto it = g_PairLoadQueue.begin(); it != g_PairLoadQueue.end();)
            {
                if (it->first == botGuid && it->second == playerGuid)
                    it = g_PairLoadQueue.erase(it);
                else
                    ++it;
            }
            {
                std::lock_guard<std::mutex> cooldownLock(g_MaintenanceFailCooldownMutex);
                if (g_MaintenanceFailCooldownUntil.count(botGuid))
                    g_MaintenanceFailCooldownUntil[botGuid].erase(playerGuid);
            }
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
        for (auto const& [botGuid, playerMap] : g_PairMemory)
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

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid))
        {
            if (!pair->playerName.empty())
                return pair->playerName;
        }
    }

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
    std::string facts;
    std::string notes;
    uint32_t wm = 0;
    uint32_t pending = 0;
    uint32_t archiveRing = 0;
    time_t lastPlayerAt = 0;
    time_t lastMaintenanceAt = 0;

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        if (PairMemoryState* pair = GetPairUnlocked(botGuid, playerGuid))
        {
            facts = pair->facts;
            notes = pair->notes;
            archiveRing = static_cast<uint32_t>(pair->archiveRing.size());
            lastPlayerAt = pair->lastPlayerAt;
            lastMaintenanceAt = pair->lastMaintenanceAt;
        }
        wm = GetWatermark(botGuid, playerGuid);
        pending = PendingCountUnlocked(botGuid, playerGuid);
    }

    uint32_t archiveRows = 0;
    std::string lastMaintenance = "never";
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
            lastMaintenance = (*lc)[0].Get<std::string>();
    }

    return fmt::format(
        "watermark={} pending={} archive_ring={} archive_rows={} maint_db={} player_at={} maint_at={}\n"
        "[FACTS]\n{}\n[NOTES]\n{}",
        wm, pending, archiveRing, archiveRows, lastMaintenance,
        lastPlayerAt ? std::to_string(lastPlayerAt) : "never",
        lastMaintenanceAt ? std::to_string(lastMaintenanceAt) : "never",
        facts.empty() ? "(empty)" : facts,
        notes.empty() ? "(empty)" : notes);
}
