#include "Log.h"
#include "Language.h"
#include "Player.h"
#include "Chat.h"
#include "Channel.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "Config.h"
#include "Common.h"
#include "Guild.h"
#include "GuildMgr.h"
#include "ObjectAccessor.h"
#include "World.h"
#include "AiFactory.h"
#include "ChannelMgr.h"
#include <sstream>
#include <vector>
#include <list>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <algorithm>
#include <random>
#include <cctype>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <mutex>
#include "DatabaseEnv.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat_sanitize.h"
#include "mod-ollama-chat_verify.h"
#include <iomanip>
#include "SpellMgr.h"
#include "SpellInfo.h"
#include "SharedDefines.h"
#include "Group.h"
#include "Creature.h"
#include "GameObject.h"
#include "TravelMgr.h"
#include "TravelNode.h"
#include "ObjectMgr.h"
#include "QuestDef.h"

// For AzerothCore range checks
#include "GridNotifiersImpl.h"
#include "CellImpl.h"
#include "Map.h"
#include "GridNotifiers.h"

// Forward declarations for internal helper functions.
static bool IsBotEligibleForChatChannelLocal(Player* bot, Player* player,
                                             ChatChannelSourceLocal source, Channel* channel = nullptr, Player* receiver = nullptr);

// Helper function to format class name for any player
static std::string FormatPlayerClass(uint8_t classId)
{
    switch (classId)
    {
        case CLASS_WARRIOR:      return "Warrior";
        case CLASS_PALADIN:      return "Paladin";
        case CLASS_HUNTER:       return "Hunter";
        case CLASS_ROGUE:        return "Rogue";
        case CLASS_PRIEST:       return "Priest";
        case CLASS_DEATH_KNIGHT: return "Death Knight";
        case CLASS_SHAMAN:       return "Shaman";
        case CLASS_MAGE:         return "Mage";
        case CLASS_WARLOCK:      return "Warlock";
        case CLASS_DRUID:        return "Druid";
        default:                 return "Unknown";
    }
}

// Helper function to format race name for any player
static std::string FormatPlayerRace(uint8_t raceId)
{
    switch (raceId)
    {
        case RACE_HUMAN:         return "Human";
        case RACE_ORC:           return "Orc";
        case RACE_DWARF:         return "Dwarf";
        case RACE_NIGHTELF:      return "Night Elf";
        case RACE_UNDEAD_PLAYER: return "Undead";
        case RACE_TAUREN:        return "Tauren";
        case RACE_GNOME:         return "Gnome";
        case RACE_TROLL:         return "Troll";
        case RACE_BLOODELF:      return "Blood Elf";
        case RACE_DRAENEI:       return "Draenei";
        default:                 return "Unknown";
    }
}

const char* ChatChannelSourceLocalStr[] =
{
    "Undefined",  // 0
    "Say",        // 1
    "Party",      // 2
    "Raid",       // 3
    "Guild",      // 4
    "Officer",    // 5
    "Yell",       // 6
    "Whisper",    // 7
    "Unknown8",   // 8
    "Unknown9",   // 9
    "Unknown10",  // 10
    "Unknown11",  // 11
    "Unknown12",  // 12
    "Unknown13",  // 13
    "Unknown14",  // 14
    "Unknown15",  // 15
    "Unknown16",  // 16
    "General"     // 17
};

std::string GetConversationEntryKey(uint64_t botGuid, uint64_t playerGuid, const std::string& playerMessage, const std::string& botReply)
{
    // Use a combination that guarantees uniqueness
    return SafeFormat("{}:{}:{}:{}", botGuid, playerGuid, playerMessage, botReply);
}

std::string rtrim(const std::string& s)
{
    const std::string whitespace = " \t\n\r,.!?;:";
    size_t end = s.find_last_not_of(whitespace);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

ChatChannelSourceLocal GetChannelSourceLocal(uint32_t type)
{
    switch (type)
    {
        case CHAT_MSG_SAY:
            return SRC_SAY_LOCAL;
        case CHAT_MSG_PARTY:
        case CHAT_MSG_PARTY_LEADER:
            return SRC_PARTY_LOCAL;
        case CHAT_MSG_RAID:
        case CHAT_MSG_RAID_LEADER:
        case CHAT_MSG_RAID_WARNING:
            return SRC_RAID_LOCAL;
        case CHAT_MSG_GUILD:
            return SRC_GUILD_LOCAL;
        case CHAT_MSG_OFFICER:
            return SRC_OFFICER_LOCAL;
        case CHAT_MSG_YELL:
            return SRC_YELL_LOCAL;
        case CHAT_MSG_WHISPER:
        case CHAT_MSG_WHISPER_FOREIGN:
        case CHAT_MSG_WHISPER_INFORM:
            return SRC_WHISPER_LOCAL;
        case CHAT_MSG_CHANNEL:
            return SRC_GENERAL_LOCAL;
        default:
            return SRC_UNDEFINED_LOCAL;
    }
}

Channel* GetValidChannel(uint32_t teamId, const std::string& channelName, Player* player)
{
    ChannelMgr* cMgr = ChannelMgr::forTeam(static_cast<TeamId>(teamId));
    Channel* channel = cMgr->GetChannel(channelName, player);
    if (!channel)
    {
        if(g_DebugEnabled)
        {
            LOG_ERROR("server.loading", "[Ollama Chat] Channel '{}' not found for team {}", channelName, teamId);
        }
    }
    return channel;
}

static void EnsureBotInZoneChannel(Player* player, char const* shortName)
{
    if (!player || !g_Enable)
        return;

    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
    if (!botAI || !botAI->IsBotAI() || !player->IsInWorld())
        return;

    ChannelMgr* cMgr = ChannelMgr::forTeam(player->GetTeamId());
    if (!cMgr)
        return;

    Channel* channel = cMgr->GetChannel(shortName, player, false);
    if (!channel || player->IsInChannel(channel))
        return;

    channel->JoinChannel(player, "");
}

void EnsureBotInGeneralChannel(Player* player)
{
    EnsureBotInZoneChannel(player, "General");
}

void EnsureBotInCityChannels(Player* player)
{
    EnsureBotInZoneChannel(player, "General");
    EnsureBotInZoneChannel(player, "Trade");
    EnsureBotInZoneChannel(player, "GuildRecruitment");
}

static void EnsureBotInChannel(Player* player, Channel* channel)
{
    if (!player || !channel || player->IsInChannel(channel))
        return;
    channel->JoinChannel(player, "");
}

void PlayerBotChatHandler::OnPlayerLogin(Player* player)
{
    EnsureBotInCityChannels(player);
}

void PlayerBotChatHandler::OnPlayerUpdateZone(Player* player, uint32 /*newZone*/, uint32 /*newArea*/)
{
    EnsureBotInCityChannels(player);
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Group* /*group*/)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Guild* /*guild*/)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Channel* channel)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, channel, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Player* receiver)
{
    // Only process if our module is enabled
    if (!g_Enable)
        return true;

    if (type == CHAT_MSG_WHISPER)
    {
        // Check if this is a valid whisper to a bot
        if (!receiver || !player || player == receiver)
            return true;

        // Check if sender is a bot - if so, don't trigger Ollama responses for bot-to-bot whispers
        PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
        if (senderAI && senderAI->IsBotAI())
        {
            return true;
        }

        PlayerbotAI* receiverAI = PlayerbotsMgr::instance().GetPlayerbotAI(receiver);
        if (!receiverAI || !receiverAI->IsBotAI())
            return true;
    }

    if (g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] OnPlayerCanUseChat called: player={}, type={}, receiver={}",
            player->GetName(), type, receiver ? receiver->GetName() : "null");
    }

    // Process the chat immediately in OnPlayerCanUseChat to prevent double processing
    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, receiver);

    // Return false to prevent the message from being processed again in OnPlayerChat
    return true;
}

namespace
{
constexpr size_t kGeneralHopCap = 256;
constexpr time_t kGeneralHopTtl = 120;

struct GeneralHopEntry
{
    time_t expiry;
};

std::unordered_map<uint32_t, GeneralHopEntry> g_generalHopMap;
std::mutex g_generalHopMutex;

uint32_t HashGeneralMsg(std::string const& msg)
{
    uint32_t h = 2166136261u;
    for (unsigned char c : msg)
    {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

void PruneGeneralHopLocked()
{
    time_t now = time(nullptr);
    for (auto it = g_generalHopMap.begin(); it != g_generalHopMap.end();)
    {
        if (difftime(now, it->second.expiry) > 0)
            it = g_generalHopMap.erase(it);
        else
            ++it;
    }
    while (g_generalHopMap.size() > kGeneralHopCap)
        g_generalHopMap.erase(g_generalHopMap.begin());
}

void MarkBotGeneralLine(std::string const& msg)
{
    if (msg.empty())
        return;
    std::lock_guard<std::mutex> lock(g_generalHopMutex);
    PruneGeneralHopLocked();
    g_generalHopMap[HashGeneralMsg(msg)] = { time(nullptr) + kGeneralHopTtl };
}

bool IsBotGeneralLine(std::string const& msg)
{
    if (msg.empty())
        return false;
    std::lock_guard<std::mutex> lock(g_generalHopMutex);
    auto it = g_generalHopMap.find(HashGeneralMsg(msg));
    if (it == g_generalHopMap.end())
        return false;
    if (difftime(time(nullptr), it->second.expiry) > 0)
        return false;
    return true;
}

bool DeliverGeneralChat(Player* bot, std::string const& line, Channel* channel)
{
    if (!bot || !channel || line.empty())
        return false;

    EnsureBotInChannel(bot, channel);
    if (!bot->IsInChannel(channel))
        return false;

    channel->Say(bot->GetGUID(), line, LANG_UNIVERSAL);
    MarkBotGeneralLine(line);
    AppendZoneGeneralTranscript(bot->GetZoneId(), bot->GetName(), line, true);
    if (HumanActiveInZoneGeneralRecently(bot->GetZoneId()))
        ProcessBotChatMessage(bot, line, SRC_GENERAL_LOCAL, channel);
    return true;
}
} // namespace

namespace
{
constexpr size_t kGeneralTranscriptCap = 12;
constexpr uint32_t kBotGeneralHumanActivityMinutes = 5;
constexpr uint32_t kGeneralMaxBotsPerHumanMessage = 2;
constexpr uint32_t kReplyVerificationMaxRetries = 1;

struct GeneralTranscriptLine
{
    std::string speaker;
    std::string text;
};

std::unordered_map<uint32_t, std::deque<GeneralTranscriptLine>> g_zoneGeneralTranscript;
std::unordered_map<uint32_t, time_t> g_lastHumanGeneralTime;
std::mutex g_zoneTranscriptMutex;

void PruneHumanActivityLocked()
{
    time_t const cutoff = time(nullptr) - static_cast<time_t>(kBotGeneralHumanActivityMinutes * 60);
    for (auto it = g_lastHumanGeneralTime.begin(); it != g_lastHumanGeneralTime.end();)
    {
        if (it->second < cutoff)
            it = g_lastHumanGeneralTime.erase(it);
        else
            ++it;
    }
}
} // namespace

void AppendZoneGeneralTranscript(uint32_t zoneId, std::string const& speaker, std::string const& text, bool /*isBot*/)
{
    if (zoneId == 0 || speaker.empty() || text.empty())
        return;
    std::lock_guard<std::mutex> lock(g_zoneTranscriptMutex);
    auto& dq = g_zoneGeneralTranscript[zoneId];
    dq.push_back({ speaker, text });
    while (dq.size() > kGeneralTranscriptCap)
        dq.pop_front();
}

std::string FormatRecentGeneralTranscript(uint32_t zoneId, size_t maxLines)
{
    std::lock_guard<std::mutex> lock(g_zoneTranscriptMutex);
    auto it = g_zoneGeneralTranscript.find(zoneId);
    if (it == g_zoneGeneralTranscript.end() || it->second.empty())
        return "";

    auto const& dq = it->second;
    size_t start = dq.size() > maxLines ? dq.size() - maxLines : 0;
    std::ostringstream ss;
    for (size_t i = start; i < dq.size(); ++i)
    {
        if (i > start)
            ss << '\n';
        ss << dq[i].speaker << ": " << dq[i].text;
    }
    return ss.str();
}

void MarkHumanGeneralActivity(uint32_t zoneId)
{
    if (zoneId == 0)
        return;
    std::lock_guard<std::mutex> lock(g_zoneTranscriptMutex);
    g_lastHumanGeneralTime[zoneId] = time(nullptr);
    PruneHumanActivityLocked();
}

static void NoteHumanGeneralChat(uint32_t zoneId, std::string const& speaker, std::string const& text)
{
    if (zoneId == 0 || speaker.empty() || text.empty())
        return;
    std::lock_guard<std::mutex> lock(g_zoneTranscriptMutex);
    auto& dq = g_zoneGeneralTranscript[zoneId];
    dq.push_back({ speaker, text });
    while (dq.size() > kGeneralTranscriptCap)
        dq.pop_front();
    g_lastHumanGeneralTime[zoneId] = time(nullptr);
    PruneHumanActivityLocked();
}

bool HumanActiveInZoneGeneralRecently(uint32_t zoneId)
{
    if (zoneId == 0)
        return false;
    std::lock_guard<std::mutex> lock(g_zoneTranscriptMutex);
    auto it = g_lastHumanGeneralTime.find(zoneId);
    if (it == g_lastHumanGeneralTime.end())
        return false;
    return difftime(time(nullptr), it->second) <= static_cast<double>(kBotGeneralHumanActivityMinutes * 60);
}

static std::vector<std::string> CollectReferenceNames(Player* bot, std::vector<Player*> const& candidates)
{
    std::vector<std::string> names;
    auto addName = [&](std::string const& n) {
        if (n.empty())
            return;
        for (std::string const& existing : names)
        {
            if (existing == n)
                return;
        }
        names.push_back(n);
    };
    if (bot)
    {
        GroupContext grp = BuildGroupContext(bot);
        for (std::string const& n : grp.memberNames)
            addName(n);
    }
    for (Player* p : candidates)
    {
        if (p)
            addName(p->GetName());
    }
    return names;
}

bool TrySendGeneralChat(Player* bot, std::string& response, std::string const& triggerMsg, Channel* channel)
{
    if (!bot || !channel || response.empty())
        return false;
    if (!FinalizeGeneralChatLine(response, triggerMsg))
        return false;
    return DeliverGeneralChat(bot, response, channel);
}

void AppendBotConversation(uint64_t botGuid, uint64_t playerGuid, const std::string& playerMessage, const std::string& botReply, bool isEvent, ChatChannelSourceLocal channel, bool senderIsBot, bool verified)
{
    if (channel == SRC_GENERAL_LOCAL && senderIsBot)
        return;

    if (g_EnableMemory)
    {
        AppendBotMemoryTurn(botGuid, playerGuid, playerMessage, botReply, isEvent, verified);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto& playerHistory = g_BotConversationHistory[botGuid][playerGuid];
        playerHistory.push_back({ playerMessage, botReply, verified });

        while (playerHistory.size() > g_MaxConversationHistory)
            playerHistory.pop_front();
    }
}

void SaveBotConversationHistoryToDB()
{
    if (g_EnableMemory)
        return;

    struct HistoryRow
    {
        uint64_t botGuid;
        uint64_t playerGuid;
        std::string playerMessage;
        std::string botReply;
    };

    std::vector<HistoryRow> rows;
    rows.reserve(256);

    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

        for (auto const& [botGuid, playerMap] : g_BotConversationHistory)
        {
            for (auto const& [playerGuid, history] : playerMap)
            {
                for (auto const& pair : history)
                {
                    rows.push_back({
                        botGuid,
                        playerGuid,
                        pair.playerMessage,
                        pair.botReply
                    });
                }
            }
        }
    }

    CharacterDatabaseTransaction trans = CharacterDatabase.BeginTransaction();
    for (auto const& row : rows)
    {
        std::string escPlayerMsg = row.playerMessage;
        CharacterDatabase.EscapeString(escPlayerMsg);

        std::string escBotReply = row.botReply;
        CharacterDatabase.EscapeString(escBotReply);

        trans->Append(SafeFormat(
            "INSERT IGNORE INTO mod_ollama_chat_history (bot_guid, player_guid, timestamp, player_message, bot_reply) "
            "VALUES ({}, {}, NOW(), '{}', '{}')",
            row.botGuid, row.playerGuid, escPlayerMsg, escBotReply));
    }
    if (trans->GetSize() > 0)
        CharacterDatabase.CommitTransaction(trans);

    // The window-function cleanup delete is much heavier than the inserts, so run
    // it at most once per hour rather than on every save.
    static time_t lastHistoryCleanup = 0;
    time_t nowCleanup = time(nullptr);
    if (lastHistoryCleanup != 0 && difftime(nowCleanup, lastHistoryCleanup) < 3600.0)
        return;
    lastHistoryCleanup = nowCleanup;

    std::string cleanupQuery = R"SQL(
        WITH ranked_history AS (
            SELECT
                bot_guid,
                player_guid,
                timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY bot_guid, player_guid
                    ORDER BY timestamp DESC
                ) as rn
            FROM mod_ollama_chat_history
        )
        DELETE FROM mod_ollama_chat_history
        WHERE (bot_guid, player_guid, timestamp) IN (
            SELECT bot_guid, player_guid, timestamp
            FROM ranked_history
            WHERE rn > {}
        );
    )SQL";
    CharacterDatabase.Execute(SafeFormat(cleanupQuery, g_MaxConversationHistory));
}

// Called when a bot sends a message (random chatter or other bot-initiated messages)
// This triggers other bots to potentially reply
void ProcessBotChatMessage(Player* bot, const std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel)
{
    if (!bot || msg.empty())
        return;
        
    // If channel is nullptr but this is a channel-type message, try to find the channel
    if (!channel && sourceLocal == SRC_GENERAL_LOCAL)
    {
        // Look up the General channel for this bot's faction
        std::string channelName = "General";
        ChannelMgr* cMgr = ChannelMgr::forTeam(bot->GetTeamId());
        if (cMgr)
        {
            channel = cMgr->GetChannel(channelName, bot);
            if (g_DebugEnabled)
            {
                if (channel)
                    LOG_INFO("server.loading", "[Ollama Chat] ProcessBotChatMessage: Found General channel for bot {}", bot->GetName());
                else
                    LOG_ERROR("server.loading", "[Ollama Chat] ProcessBotChatMessage: Could not find General channel for bot {}", bot->GetName());
            }
        }
    }
    
    // Validate that bot is actually in the relevant chat group before triggering replies
    bool canSendMessage = false;
    switch (sourceLocal)
    {
        case SRC_SAY_LOCAL:
        case SRC_YELL_LOCAL:
            // Distance checks will be applied during eligibility filtering
            canSendMessage = true;
            break;
            
        case SRC_GENERAL_LOCAL:
            // Must have a channel object
            canSendMessage = (channel != nullptr);
            if (!canSendMessage && g_DebugEnabled)
                LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to General - no channel found", bot->GetName());
            break;
            
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            // Must be in a guild with at least one real player online
            if (bot->GetGuildId() != 0)
            {
                Guild* guild = sGuildMgr->GetGuildById(bot->GetGuildId());
                if (guild)
                {
                    // Check if any real (non-bot) players are online in this guild
                    bool hasRealPlayer = false;
                    for (auto const& pair : ObjectAccessor::GetPlayers())
                    {
                        Player* member = pair.second;
                        if (member && member->GetGuildId() == bot->GetGuildId())
                        {
                            if (!PlayerbotsMgr::instance().GetPlayerbotAI(member))
                            {
                                hasRealPlayer = true;
                                break;
                            }
                        }
                    }
                    canSendMessage = hasRealPlayer;
                    if (!canSendMessage && g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - no real players online in guild", bot->GetName());
                }
                else
                {
                    canSendMessage = false;
                    if (g_DebugEnabled)
                        LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - guild not found", bot->GetName());
                }
            }
            else
            {
                canSendMessage = false;
                if (g_DebugEnabled)
                    LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - not in a guild", bot->GetName());
            }
            break;
            
        case SRC_PARTY_LOCAL:
        case SRC_RAID_LOCAL:
            // Must be in a group with at least one real player
            if (bot->GetGroup())
            {
                Group* group = bot->GetGroup();
                bool hasRealPlayer = false;
                for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
                {
                    Player* member = ref->GetSource();
                    if (member && !PlayerbotsMgr::instance().GetPlayerbotAI(member))
                    {
                        hasRealPlayer = true;
                        break;
                    }
                }
                canSendMessage = hasRealPlayer;
                if (!canSendMessage && g_DebugEnabled)
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} cannot send to Party - no real players in group", bot->GetName());
            }
            else
            {
                canSendMessage = false;
                if (g_DebugEnabled)
                    LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Party - not in a group", bot->GetName());
            }
            break;
            
        case SRC_WHISPER_LOCAL:
            // Whispers are handled separately
            canSendMessage = true;
            break;
            
        default:
            canSendMessage = true;
            break;
    }
    
    if (!canSendMessage)
    {
        if (g_DebugEnabled)
            LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send message to {} - validation failed", 
                    bot->GetName(), ChatChannelSourceLocalStr[sourceLocal]);
        return;
    }
        
    // Convert ChatChannelSourceLocal back to chat type for ProcessChat
    uint32_t type = 0;
    switch (sourceLocal)
    {
        case SRC_SAY_LOCAL: type = CHAT_MSG_SAY; break;
        case SRC_YELL_LOCAL: type = CHAT_MSG_YELL; break;
        case SRC_PARTY_LOCAL: type = CHAT_MSG_PARTY; break;
        case SRC_RAID_LOCAL: type = CHAT_MSG_RAID; break;
        case SRC_GUILD_LOCAL: type = CHAT_MSG_GUILD; break;
        case SRC_OFFICER_LOCAL: type = CHAT_MSG_OFFICER; break;
        case SRC_WHISPER_LOCAL: type = CHAT_MSG_WHISPER; break;
        case SRC_GENERAL_LOCAL: type = CHAT_MSG_CHANNEL; break;
        default: type = CHAT_MSG_SAY; break;
    }
    
    std::string mutableMsg = msg; // ProcessChat takes non-const reference
    uint32_t lang = bot->GetTeamId() == TEAM_ALLIANCE ? LANG_COMMON : LANG_ORCISH;
    
    // Call the main ProcessChat function with bot as sender
    PlayerBotChatHandler::ProcessChat(bot, type, lang, mutableMsg, sourceLocal, channel, nullptr);
}

std::string GetBotHistoryPrompt(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage)
{
    if (!g_EnableChatHistory)
        return "";

    std::deque<ConversationTurn> historyCopy;
    {
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

        const auto botIt = g_BotConversationHistory.find(botGuid);
        if (botIt == g_BotConversationHistory.end())
            return "";
        const auto playerIt = botIt->second.find(playerGuid);
        if (playerIt == botIt->second.end())
            return "";

        size_t startIdx = 0;
        if (g_EnableMemory)
        {
            uint32_t wm = 0;
            auto cit = g_CompactedTurnCount.find(botGuid);
            if (cit != g_CompactedTurnCount.end())
            {
                auto pit = cit->second.find(playerGuid);
                if (pit != cit->second.end())
                    wm = pit->second;
            }
            startIdx = std::min(static_cast<size_t>(wm), playerIt->second.size());
        }
        for (size_t i = startIdx; i < playerIt->second.size(); ++i)
            historyCopy.push_back(playerIt->second[i]);
    }

    Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
    std::string playerName = player ? player->GetName() : "The player";

    std::string result;
    result += SafeFormat(g_ChatHistoryHeaderTemplate, fmt::arg("player_name", playerName));

    for (const auto& entry : historyCopy)
    {
        result += SafeFormat(g_ChatHistoryLineTemplate,
            fmt::arg("player_name", playerName),
            fmt::arg("player_message", entry.playerMessage),
            fmt::arg("bot_reply", entry.botReply)
        );
    }

    result += SafeFormat(g_ChatHistoryFooterTemplate,
        fmt::arg("player_name", playerName),
        fmt::arg("player_message", playerMessage)
    );

    return result;
}

// --- Helper: Spells ---
std::string ChatHandler_GetBotSpellInfo(Player* bot)
{
    // Map to store highest rank of each spell: spell name -> (spellId, rank, costText)
    std::map<std::string, std::tuple<uint32, uint32, std::string>> uniqueSpells;
    
    for (const auto& spellPair : bot->GetSpellMap())
    {
        uint32 spellId = spellPair.first;
        const SpellInfo* spellInfo = sSpellMgr->GetSpellInfo(spellId);
        if (!spellInfo || spellInfo->Attributes & SPELL_ATTR0_PASSIVE)
            continue;
        if (spellInfo->SpellFamilyName == SPELLFAMILY_GENERIC)
            continue;
        if (bot->HasSpellCooldown(spellId))
            continue;
        
        const char* name = spellInfo->SpellName[0];
        if (!name || !*name)
            continue;
        
        std::string costText;
        if (spellInfo->ManaCost || spellInfo->ManaCostPercentage)
        {
            switch (spellInfo->PowerType)
            {
                case POWER_MANA: costText = std::to_string(spellInfo->ManaCost) + " mana"; break;
                case POWER_RAGE: costText = std::to_string(spellInfo->ManaCost) + " rage"; break;
                case POWER_FOCUS: costText = std::to_string(spellInfo->ManaCost) + " focus"; break;
                case POWER_ENERGY: costText = std::to_string(spellInfo->ManaCost) + " energy"; break;
                case POWER_RUNIC_POWER: costText = std::to_string(spellInfo->ManaCost) + " runic power"; break;
                default: costText = std::to_string(spellInfo->ManaCost) + " unknown resource"; break;
            }
        }
        else
        {
            costText = "no cost";
        }
        
        // Get base spell name (without rank)
        std::string spellName = name;
        uint32 rank = spellInfo->GetRank();
        
        // Check if we already have this spell, and if so, only keep the highest rank
        auto it = uniqueSpells.find(spellName);
        if (it == uniqueSpells.end())
        {
            // First time seeing this spell
            uniqueSpells[spellName] = std::make_tuple(spellId, rank, costText);
        }
        else
        {
            // We've seen this spell before, check if this is a higher rank
            uint32 existingRank = std::get<1>(it->second);
            if (rank > existingRank)
            {
                // Replace with higher rank
                uniqueSpells[spellName] = std::make_tuple(spellId, rank, costText);
            }
        }
    }
    
    // Build the output string from unique spells
    std::ostringstream spellSummary;
    for (const auto& [spellName, spellData] : uniqueSpells)
    {
        uint32 rank = std::get<1>(spellData);
        const std::string& costText = std::get<2>(spellData);
        
        spellSummary << "**" << spellName << "**";
        if (rank > 0)
        {
            spellSummary << " (Rank " << rank << ")";
        }
        spellSummary << " - Costs " << costText << "\n";
    }
    return spellSummary.str();
}

// --- Helper: Group info ---
std::vector<std::string> ChatHandler_GetGroupStatus(Player* bot)
{
    std::vector<std::string> info;
    if (!bot || !bot->GetGroup()) return info;
    Group* group = bot->GetGroup();
    for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
    {
        Player* member = ref->GetSource();
        if (!member || !member->GetMap()) continue;
        if(bot == member) continue;
        float dist = bot->GetDistance(member);
        std::string beingAttacked = "";
        if (Unit* attacker = member->GetVictim())
        {
            beingAttacked = " [Under Attack by " + attacker->GetName() +
                            ", Level: " + std::to_string(attacker->GetLevel()) + ", HP: " + std::to_string(attacker->GetHealth()) +
                            "/" + std::to_string(attacker->GetMaxHealth()) + ")]";
        }
        std::string className = FormatPlayerClass(member->getClass());
        std::string raceName = FormatPlayerRace(member->getRace());
        std::string memberType = PlayerbotsMgr::instance().GetPlayerbotAI(member) ? "bot" : "player";
        info.push_back(
            member->GetName() +
            " (Level: " + std::to_string(member->GetLevel()) +
            ", Class: " + className +
            ", Race: " + raceName +
            ", HP: " + std::to_string(member->GetHealth()) + "/" + std::to_string(member->GetMaxHealth()) +
            ", Dist: " + std::to_string(dist) + ", " + memberType + ")" + beingAttacked
        );

    }
    return info;
}

GroupContext BuildGroupContext(Player* bot)
{
    GroupContext ctx;
    Group* group = bot ? bot->GetGroup() : nullptr;
    if (!group)
    {
        ctx.statusLine = "Solo";
        return ctx;
    }

    std::ostringstream party, roster;
    bool rosterStarted = false;

    for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
    {
        Player* member = ref->GetSource();
        if (!member || !member->GetMap())
            continue;

        ++ctx.memberCount;
        bool isBot = PlayerbotsMgr::instance().GetPlayerbotAI(member) != nullptr;
        if (isBot)
            ++ctx.botCount;
        else
            ++ctx.playerCount;

        if (member == bot)
            continue;

        std::string line = fmt::format("{} (L{} {}, {})", member->GetName(), member->GetLevel(),
            FormatPlayerClass(member->getClass()), isBot ? "bot" : "player");
        party << "- " << line << "\n";
        ctx.memberNames.push_back(member->GetName());
        if (rosterStarted)
            roster << ", ";
        roster << line;
        rosterStarted = true;
    }

    if (!ctx.memberCount)
    {
        ctx.statusLine = "Solo";
        return ctx;
    }

    bool duo = ctx.memberCount == 2;
    ctx.statusLine = duo
        ? fmt::format("2-member party: only you and {}", roster.str())
        : fmt::format("Party of {}: {}", ctx.memberCount, roster.str());

    if (duo)
        party << "Only 2 people in this party. Do not say everyone or refer to other teammates.";

    ctx.partySection = party.str();
    if (ctx.partySection.size() > 200)
        ctx.partySection.resize(200);
    return ctx;
}

// --- Helper: Visible players ---
std::vector<std::string> ChatHandler_GetVisiblePlayers(Player* bot, float radius)
{
    std::vector<std::string> players;
    if (!bot || !bot->GetMap())
        return players;

    Map* map = bot->GetMap();
    for (auto const& ref : map->GetPlayers())
    {
        Player* player = ref.GetSource();
        if (!player || player == bot)
            continue;
        if (!player->IsInWorld() || player->IsGameMaster())
            continue;
        if (!bot->IsWithinDistInMap(player, radius))
            continue;
        if (!bot->IsWithinLOS(player->GetPositionX(), player->GetPositionY(), player->GetPositionZ()))
            continue;

        float dist = bot->GetDistance(player);
        std::string faction = (player->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde");
        std::string className = FormatPlayerClass(player->getClass());
        std::string raceName = FormatPlayerRace(player->getRace());
        players.push_back(
            "Player: " + player->GetName() +
            " (Level: " + std::to_string(player->GetLevel()) +
            ", Class: " + className +
            ", Race: " + raceName +
            ", Faction: " + faction +
            ", Distance: " + std::to_string(dist) + ")"
        );
        if (players.size() >= 6)
            break;
    }
    return players;
}

// --- Helper: Visible locations/objects (creatures and gameobjects) ---
std::vector<std::string> ChatHandler_GetVisibleLocations(Player* bot, float radius)
{
    std::vector<std::string> visible;
    if (!bot || !bot->GetMap())
        return visible;

    std::list<Unit*> units;
    Acore::AnyUnitInObjectRangeCheck unitCheck(bot, radius);
    Acore::UnitListSearcher<Acore::AnyUnitInObjectRangeCheck> unitSearcher(bot, units, unitCheck);
    Cell::VisitObjects(bot, unitSearcher, radius);

    for (Unit* unit : units)
    {
        if (visible.size() >= 6)
            break;
        if (!unit || unit->GetTypeId() != TYPEID_UNIT)
            continue;

        Creature* c = unit->ToCreature();
        if (!c || c->GetGUID() == bot->GetGUID())
            continue;
        if (c->IsPet() || c->IsTotem())
            continue;
        if (!bot->IsWithinLOS(c->GetPositionX(), c->GetPositionY(), c->GetPositionZ()))
            continue;

        std::string type;
        if (c->isDead())
            type = "DEAD";
        else if (c->IsHostileTo(bot))
            type = "ENEMY";
        else if (c->IsFriendlyTo(bot))
            type = "FRIENDLY";
        else
            type = "NEUTRAL";

        float dist = bot->GetDistance(c);
        visible.push_back(
            type + ": " + c->GetName() +
            ", Level: " + std::to_string(c->GetLevel()) +
            ", HP: " + std::to_string(c->GetHealth()) + "/" + std::to_string(c->GetMaxHealth()) +
            ", Distance: " + std::to_string(dist) + ")"
        );
    }

    std::list<GameObject*> gameObjects;
    Acore::GameObjectInRangeCheck goCheck(bot->GetPositionX(), bot->GetPositionY(), bot->GetPositionZ(), radius);
    Acore::GameObjectListSearcher<Acore::GameObjectInRangeCheck> goSearcher(bot, gameObjects, goCheck);
    Cell::VisitObjects(bot, goSearcher, radius);

    for (GameObject* go : gameObjects)
    {
        if (visible.size() >= 8)
            break;
        if (!go || !bot->IsWithinLOS(go->GetPositionX(), go->GetPositionY(), go->GetPositionZ()))
            continue;

        float dist = bot->GetDistance(go);
        visible.push_back(
            go->GetName() +
            ", Type: " + std::to_string(go->GetGoType()) +
            ", Distance: " + std::to_string(dist) + ")"
        );
    }

    return visible;
}

// --- Helper: Combat summary ---
std::string ChatHandler_GetCombatSummary(Player* bot)
{
    std::ostringstream oss;
    bool inCombat = bot->IsInCombat();
    Unit* victim = bot->GetVictim();

    // Class-specific resource reporting
    auto classId = bot->getClass();

    auto printResource = [&](std::ostringstream& oss) {
        switch (classId)
        {
            case CLASS_WARRIOR:
                oss << ", Rage: " << bot->GetPower(POWER_RAGE) << "/" << bot->GetMaxPower(POWER_RAGE);
                break;
            case CLASS_ROGUE:
                oss << ", Energy: " << bot->GetPower(POWER_ENERGY) << "/" << bot->GetMaxPower(POWER_ENERGY);
                break;
            case CLASS_DEATH_KNIGHT:
                oss << ", Runic Power: " << bot->GetPower(POWER_RUNIC_POWER) << "/" << bot->GetMaxPower(POWER_RUNIC_POWER);
                break;
            case CLASS_HUNTER:
                oss << ", Focus: " << bot->GetPower(POWER_FOCUS) << "/" << bot->GetMaxPower(POWER_FOCUS);
                break;
            default: // Mana classes
                if (bot->GetMaxPower(POWER_MANA) > 0)
                    oss << ", Mana: " << bot->GetPower(POWER_MANA) << "/" << bot->GetMaxPower(POWER_MANA);
                break;
        }
    };

    if (inCombat)
    {
        oss << "IN COMBAT: ";
        if (victim)
        {
            oss << "Target: " << victim->GetName()
                << ", Level: " << victim->GetLevel()
                << ", HP: " << victim->GetHealth() << "/" << victim->GetMaxHealth();
        }
        else
        {
            oss << "No current target";
        }
        oss << ". ";
        printResource(oss);
    }
    else
    {
        oss << "NOT IN COMBAT. ";
        printResource(oss);
    }
    return oss.str();
}

std::string BuildBotPromptContext(Player* bot)
{
    if (!bot)
        return "";

    constexpr size_t kMaxBytes = 600;
    std::ostringstream ss;

    if (Map* map = bot->GetMap())
    {
        if (map->IsDungeon())
            ss << "Map: dungeon " << map->GetMapName();
        else if (map->IsRaid())
            ss << "Map: raid " << map->GetMapName();
        else if (map->IsBattleground())
            ss << "Map: battleground " << map->GetMapName();
        else
            ss << "Map: open world " << map->GetMapName();
    }

    ss << " | In combat: " << (bot->IsInCombat() ? "yes" : "no");
    if (bot->IsInCombat())
    {
        if (Unit* victim = bot->GetVictim())
            ss << " vs " << victim->GetName();
    }

    uint32 zoneId = bot->GetZoneId();
    std::string zoneQuest;
    std::string fallbackQuest;
    std::string completeQuest;
    for (auto const& [questId, qsd] : bot->getQuestStatusMap())
    {
        Quest const* quest = sObjectMgr->GetQuestTemplate(questId);
        if (!quest)
            continue;
        if (qsd.Status == QUEST_STATUS_COMPLETE && completeQuest.empty())
            completeQuest = "\"" + quest->GetTitle() + "\" ready to turn in";
        else if (qsd.Status == QUEST_STATUS_INCOMPLETE)
        {
            if (zoneQuest.empty() && quest->GetZoneOrSort() == zoneId)
                zoneQuest = "\"" + quest->GetTitle() + "\" in progress";
            else if (fallbackQuest.empty())
                fallbackQuest = "\"" + quest->GetTitle() + "\" in progress";
        }
    }
    if (zoneQuest.empty())
        zoneQuest = fallbackQuest;

    if (!zoneQuest.empty() || !completeQuest.empty())
    {
        ss << "\nQuests: ";
        if (!zoneQuest.empty())
            ss << zoneQuest;
        if (!zoneQuest.empty() && !completeQuest.empty())
            ss << "; ";
        if (!completeQuest.empty())
            ss << completeQuest;
    }

    uint32 level = bot->GetLevel();
    ss << "\nLevel " << level << " " << (bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde") << ". ";
    if (level < 20)
        ss << "No raids, heroics, or Northrend yet. ";
    else if (level < 58)
        ss << "No raids, heroics, Northrend, or ICC yet. ";
    else if (level < 70)
        ss << "No raids or ICC yet. ";
    else if (level < 80)
        ss << "No ICC hardmodes yet. ";

    ss << "OK to mention enemy cities or zones; don't say you're running or grouping for dungeons your faction doesn't use.";

    std::string result = ss.str();
    if (result.size() > kMaxBytes)
        result.resize(kMaxBytes);
    return result;
}

std::string GenerateBotGameStateSnapshot(Player* bot, bool omitGroup)
{
    // Prepare each section
    std::string combat = ChatHandler_GetCombatSummary(bot);

    std::string group;
    if (!omitGroup)
    {
        std::vector<std::string> groupInfo = ChatHandler_GetGroupStatus(bot);
        if (!groupInfo.empty())
        {
            group += "Group members:\n";
            for (const auto& entry : groupInfo)
                group += " - " + entry + "\n";
        }
    }

    std::string spells = ChatHandler_GetBotSpellInfo(bot);

    std::string quests;
    for (auto const& [questId, qsd] : bot->getQuestStatusMap())
    {
        // look up the template
        Quest const* quest = sObjectMgr->GetQuestTemplate(questId);
        if (!quest)
            continue;

        // get the English title as a fallback
        std::string title = quest->GetTitle();

        // then, if we have a locale record, overwrite it
        if (auto const* locale = sObjectMgr->GetQuestLocale(questId))
        {
            int locIdx = bot->GetSession()->GetSessionDbLocaleIndex();
            if (locIdx >= 0)
                ObjectMgr::GetLocaleString(locale->Title, locIdx, title);
        }

        // Convert quest status to readable string
        std::string statusText;
        switch (qsd.Status)
        {
            case QUEST_STATUS_NONE:       statusText = "not started"; break;
            case QUEST_STATUS_COMPLETE:   statusText = "complete (ready to turn in)"; break;
            case QUEST_STATUS_INCOMPLETE: statusText = "in progress"; break;
            case QUEST_STATUS_FAILED:     statusText = "failed"; break;
            case QUEST_STATUS_REWARDED:   statusText = "completed and rewarded"; break;
            default:                      statusText = "unknown"; break;
        }

        quests += "Quest \"" + title + "\" is " + statusText + "\n";
    }

    std::string los;
    std::vector<std::string> losLocs = ChatHandler_GetVisibleLocations(bot);
    if (!losLocs.empty()) {
        for (const auto& entry : losLocs) los += " - " + entry + "\n";
    }

    std::string players;
    std::vector<std::string> nearbyPlayers = ChatHandler_GetVisiblePlayers(bot);
    if (!nearbyPlayers.empty()) {
        for (const auto& entry : nearbyPlayers) players += " - " + entry + "\n";
    }

    // Use template
    return SafeFormat(
        g_ChatBotSnapshotTemplate,
        fmt::arg("combat", combat),
        fmt::arg("group", group),
        fmt::arg("spells", spells),
        fmt::arg("quests", quests),
        fmt::arg("los", los),
        fmt::arg("players", players)
    );
}

std::string GenerateBotPrompt(Player* bot, std::string const& playerMessage, Player* player)
{
    if (!bot || !player || g_ChatPromptTemplate.empty())
        return "";

    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!botAI || !botAI->GetChatHelper())
        return "";

    AreaTableEntry const* botCurrentArea = botAI->GetCurrentArea();
    AreaTableEntry const* botCurrentZone = botAI->GetCurrentZone();
    uint64_t botGuid = bot->GetGUID().GetRawValue();
    uint64_t playerGuid = player->GetGUID().GetRawValue();

    std::string personality = GetBotPersonality(bot);
    std::string personalityPrompt = GetPersonalityPromptAddition(personality);
    std::string botName = bot->GetName();
    uint32_t botLevel = bot->GetLevel();
    std::string botAreaName = botCurrentArea ? botAI->GetLocalizedAreaName(botCurrentArea) : "UnknownArea";
    std::string botZoneName = botCurrentZone ? botAI->GetLocalizedAreaName(botCurrentZone) : "UnknownZone";
    std::string botMapName = bot->GetMap() ? bot->GetMap()->GetMapName() : "UnknownMap";
    std::string botClass = botAI->GetChatHelper()->FormatClass(bot->getClass());
    std::string botRace = botAI->GetChatHelper()->FormatRace(bot->getRace());
    std::string botRole = ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot));
    std::string botGender = bot->getGender() == GENDER_MALE ? "Male" : "Female";
    std::string botFaction = bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde";
    std::string botGuild = bot->GetGuild() ? bot->GetGuild()->GetName() : "No Guild";
    GroupContext groupCtx = BuildGroupContext(bot);
    std::string botGroupStatus = groupCtx.statusLine;
    uint32_t botGold = bot->GetMoney() / 10000;

    std::string playerName = player->GetName();
    uint32_t playerLevel = player->GetLevel();
    std::string playerClass = botAI->GetChatHelper()->FormatClass(player->getClass());
    std::string playerRace = botAI->GetChatHelper()->FormatRace(player->getRace());
    std::string playerRole = ChatHelper::FormatClass(player, AiFactory::GetPlayerSpecTab(player));
    std::string playerGender = player->getGender() == GENDER_MALE ? "Male" : "Female";
    std::string playerFaction = player->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde";
    std::string playerGuild = player->GetGuild() ? player->GetGuild()->GetName() : "No Guild";
    std::string playerGroupStatus = groupCtx.statusLine;
    uint32_t playerGold = player->GetMoney() / 10000;
    float playerDistance = player->IsInWorld() && bot->IsInWorld() ? player->GetDistance(bot) : -1.0f;

    std::string chatHistory = GetBotHistoryPrompt(botGuid, playerGuid, playerMessage);
    std::string sentimentInfo = GetSentimentPromptAddition(bot, player);

    std::string ragInfo;
    if (g_EnableRAG && g_RAGSystem)
    {
        auto ragResults = g_RAGSystem->RetrieveRelevantInfo(
            playerMessage, g_RAGMaxRetrievedItems, g_RAGSimilarityThreshold);
        std::string ragContent = g_RAGSystem->GetFormattedRAGInfo(ragResults);
        if (!ragContent.empty() && !g_RAGPromptTemplate.empty())
            ragInfo = SafeFormat(g_RAGPromptTemplate, fmt::arg("rag_info", ragContent));
    }

    std::string extraInfo = SafeFormat(
        g_ChatExtraInfoTemplate,
        fmt::arg("bot_race", botRace),
        fmt::arg("bot_gender", botGender),
        fmt::arg("bot_role", botRole),
        fmt::arg("bot_faction", botFaction),
        fmt::arg("bot_guild", botGuild),
        fmt::arg("bot_group_status", botGroupStatus),
        fmt::arg("bot_gold", botGold),
        fmt::arg("player_race", playerRace),
        fmt::arg("player_gender", playerGender),
        fmt::arg("player_role", playerRole),
        fmt::arg("player_faction", playerFaction),
        fmt::arg("player_guild", playerGuild),
        fmt::arg("player_group_status", playerGroupStatus),
        fmt::arg("player_gold", playerGold),
        fmt::arg("player_distance", playerDistance),
        fmt::arg("bot_area", botAreaName),
        fmt::arg("bot_zone", botZoneName),
        fmt::arg("bot_map", botMapName));

    std::string prompt = SafeFormat(
        g_ChatPromptTemplate,
        fmt::arg("bot_name", botName),
        fmt::arg("bot_level", botLevel),
        fmt::arg("bot_class", botClass),
        fmt::arg("bot_personality", personalityPrompt),
        fmt::arg("bot_personality_name", personality),
        fmt::arg("player_level", playerLevel),
        fmt::arg("player_class", playerClass),
        fmt::arg("player_name", playerName),
        fmt::arg("player_message", playerMessage),
        fmt::arg("extra_info", extraInfo),
        fmt::arg("chat_history", chatHistory),
        fmt::arg("sentiment_info", sentimentInfo));

    if (!ragInfo.empty())
        prompt += ragInfo + "\n";

    if (g_EnableMemory)
    {
        std::string memory = GetMemoryPromptAddition(botGuid, playerGuid, SRC_UNDEFINED_LOCAL, playerMessage, playerName);
        if (!memory.empty())
            prompt += memory + "\n";
    }

    return prompt;
}


void PlayerBotChatHandler::ProcessChat(Player* player, uint32_t /*type*/, uint32_t lang, std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel, Player* receiver)
{
    if (player == nullptr) {
        LOG_ERROR("server.loading", "[Ollama Chat] ProcessChat: player is null");
        return;
    }
    if (msg.empty()) {
        return;
    }
    if (lang == LANG_ADDON) return;
    std::string chanName = (channel != nullptr) ? channel->GetName() : "Unknown";
    uint32_t channelId = (channel != nullptr) ? channel->GetChannelId() : 0;
    std::string receiverName = (receiver != nullptr) ? receiver->GetName() : "None";
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading",
                "[Ollama Chat] Player {} sent msg: '{}' | Source: {} | Channel Name: {} | Channel ID: {} | Receiver: {}",
                player->GetName(), msg, (int)sourceLocal, chanName, channelId, receiverName);
    }


    auto startsWithWord = [](const std::string& text, const std::string& word) {
        if (text.size() < word.size()) return false;
        if (text.compare(0, word.size(), word) != 0) return false;
        // If exact length match or next char is whitespace/punctuation, it's a word
        return text.size() == word.size() || !std::isalnum((unsigned char)text[word.size()]);
    };

    std::string trimmedMsg = rtrim(msg);
    for (const std::string& blacklist : g_BlacklistCommands)
    {
        if (startsWithWord(trimmedMsg, blacklist))
        {
            if (g_DebugEnabled)
                LOG_INFO("server.loading",
                         "[Ollama Chat] Message starts with '{}' (blacklisted). Skipping bot responses.",
                         blacklist);
            return;
        }
    }
    
    // Check if this channel type is disabled
    if (sourceLocal == SRC_GENERAL_LOCAL && g_DisableForCustomChannels)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Custom channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL) && g_DisableForSayYell)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Say/Yell channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL) && g_DisableForGuild)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Guild channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL) && g_DisableForParty)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Party/Raid channels are disabled, skipping");
        }
        return;
    }
             
    PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
    bool senderIsBot = (senderAI && senderAI->IsBotAI());

    if (senderIsBot && sourceLocal == SRC_GENERAL_LOCAL && IsBotGeneralLine(msg))
        return;

    if (!senderIsBot && sourceLocal == SRC_GENERAL_LOCAL)
        NoteHumanGeneralChat(player->GetZoneId(), player->GetName(), msg);

    if (senderIsBot && sourceLocal == SRC_GENERAL_LOCAL &&
        !HumanActiveInZoneGeneralRecently(player->GetZoneId()))
        return;
    
    std::vector<Player*> eligibleBots;
    
    // Handle different chat sources differently
    if (sourceLocal == SRC_WHISPER_LOCAL && receiver != nullptr)
    {
        // Check if whisper replies are disabled
        if (!g_EnableWhisperReplies)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Whisper replies are disabled, skipping");
            }
            return;
        }
        
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Processing whisper from {} to {}", 
                    player->GetName(), receiver->GetName());
        }
        
        // Skip bot-to-bot whispers to prevent Ollama responses
        if (senderIsBot)
        {
            return;
        }
        
        // For whispers, only the receiver bot can respond (if it's a bot)
        PlayerbotAI* receiverAI = PlayerbotsMgr::instance().GetPlayerbotAI(receiver);
        if (receiverAI && receiverAI->IsBotAI())
        {
            eligibleBots.push_back(receiver);
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Found eligible bot {} for whisper", receiver->GetName());
            }
        }
        else if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Whisper target {} is not a bot or has no AI", receiver->GetName());
        }
    }
    else if (channel != nullptr)
    {
        // For channel chat, find all bots that are in the same channel instance
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Processing channel message in '{}' (ID: {})", 
                    channel->GetName(), channel->GetChannelId());
        }
        
        // For channel chat, simply find all bots in the same zone as the player
        auto const& allPlayers = ObjectAccessor::GetPlayers();
        for (auto const& itr : allPlayers)
        {
            Player* candidate = itr.second;
            if (!candidate || candidate == player)
                continue;
                
            // Skip non-bots early
            PlayerbotAI* candidateAI = PlayerbotsMgr::instance().GetPlayerbotAI(candidate);
            if (!candidateAI || !candidateAI->IsBotAI())
                continue;
            
            // Check if this is a local or global channel
            bool isLocalChannel = (channel->GetName().find("General -") != std::string::npos ||
                                  channel->GetName().find("Trade -") != std::string::npos ||
                                  channel->GetName().find("GuildRecruitment -") != std::string::npos ||
                                  channel->GetName().find("LocalDefense -") != std::string::npos);
            
            bool isGlobalChannel = (channel->GetName().find("World") != std::string::npos || channel->GetName().find("LookingForGroup") != std::string::npos);
        
            // For local channels, bot must be in same zone as player
            if (isLocalChannel)
            {
                // ZONE CHECK: Bot must be in exact same zone as player
                if (candidate->GetZoneId() != player->GetZoneId())
                {
                    if(g_DebugEnabled)
                    {
                        //LOG_ERROR("server.loading", "[Ollama Chat] Bot {} FAILED zone check - Bot zone: {}, Player zone: {}, Channel: '{}'", candidate->GetName(), candidate->GetZoneId(), player->GetZoneId(), channel->GetName());
                    }
                    continue; // SKIP this bot - wrong zone
                }
            }
            // For global channels like World, no zone restriction
            
            // CHANNEL MEMBERSHIP CHECK: Bot must actually be in the channel
            if (!candidate->IsInChannel(channel))
            {
                if (isLocalChannel && candidate->GetZoneId() == player->GetZoneId())
                    channel->JoinChannel(candidate, "");
            }
            if (!candidate->IsInChannel(channel))
            {
                if(g_DebugEnabled)
                {
                    //LOG_INFO("server.loading", "[Ollama Chat] Bot {} not in channel '{}', skipping", candidate->GetName(), channel->GetName());
                }
                continue;
            }
            
            // FACTION CHECK: For non-global channels, ensure same faction
            if (candidate->GetTeamId() != player->GetTeamId())
            {
                if (!isGlobalChannel)
                {
                    if(g_DebugEnabled)
                    {
                        //LOG_ERROR("server.loading", "[Ollama Chat] Bot {} FAILED faction check - Bot: {}, Player: {}, Channel: '{}'", candidate->GetName(), (int)candidate->GetTeamId(), (int)player->GetTeamId(), channel->GetName());
                    }
                    continue; // SKIP this bot - wrong faction
                }
            }
            
            // REAL PLAYER CHECK: Channel must have at least one real player
            bool hasRealPlayerInChannel = false;
            for (auto const& playerItr : allPlayers)
            {
                Player* potentialRealPlayer = playerItr.second;
                if (potentialRealPlayer && potentialRealPlayer->IsInChannel(channel))
                {
                    PlayerbotAI* realPlayerAI = PlayerbotsMgr::instance().GetPlayerbotAI(potentialRealPlayer);
                    if (!realPlayerAI || !realPlayerAI->IsBotAI())
                    {
                        hasRealPlayerInChannel = true;
                        break;
                    }
                }
            }
            
            if (!hasRealPlayerInChannel)
            {
                if(g_DebugEnabled)
                {
                    //LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipped - no real players in channel '{}'", candidate->GetName(), channel->GetName());
                }
                continue;
            }
            
            // ONLY add bots that passed ALL verifications
            eligibleBots.push_back(candidate);
            if(g_DebugEnabled)
            {
                // LOG_INFO("server.loading", "[Ollama Chat] VERIFIED eligible bot {} in channel '{}' - Distance: {:.2f}, Zone match: {}", candidate->GetName(), channel->GetName(), candidate->GetDistance(player), (candidate->GetZoneId() == player->GetZoneId()));
            }
        }
        
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Found {} bots in channel instance '{}'", 
                    eligibleBots.size(), channel->GetName());
        }
    }
    else
    {
        // For other chat types (say, yell, guild, party, etc.), use all players and filter by eligibility
        auto const& allPlayers = ObjectAccessor::GetPlayers();
        for (auto const& itr : allPlayers)
        {
            Player* candidate = itr.second;
            if (candidate->IsInWorld() && candidate != player)
            {
                PlayerbotAI* candidateAI = PlayerbotsMgr::instance().GetPlayerbotAI(candidate);
                if (candidateAI && candidateAI->IsBotAI())
                {
                    // For Guild/Party, verify there's a real player in that guild/party
                    if (sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL)
                    {
                        if (candidate->GetGuildId() != 0)
                        {
                            // Check if any real player is online in this guild
                            bool hasRealPlayerInGuild = false;
                            for (auto const& guildPlayerItr : allPlayers)
                            {
                                Player* guildMember = guildPlayerItr.second;
                                if (guildMember && guildMember->GetGuildId() == candidate->GetGuildId())
                                {
                                    PlayerbotAI* memberAI = PlayerbotsMgr::instance().GetPlayerbotAI(guildMember);
                                    if (!memberAI || !memberAI->IsBotAI())
                                    {
                                        hasRealPlayerInGuild = true;
                                        break;
                                    }
                                }
                            }
                            if (!hasRealPlayerInGuild)
                                continue; // Skip bot - no real players in guild
                        }
                    }
                    else if (sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL)
                    {
                        Group* group = candidate->GetGroup();
                        if (group)
                        {
                            // Check if any real player is in this group
                            bool hasRealPlayerInGroup = false;
                            for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
                            {
                                Player* member = ref->GetSource();
                                if (member)
                                {
                                    PlayerbotAI* memberAI = PlayerbotsMgr::instance().GetPlayerbotAI(member);
                                    if (!memberAI || !memberAI->IsBotAI())
                                    {
                                        hasRealPlayerInGroup = true;
                                        break;
                                    }
                                }
                            }
                            if (!hasRealPlayerInGroup)
                                continue; // Skip bot - no real players in group
                        }
                    }
                    else if (sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL)
                    {
                        // For Say/Yell, require a real player within hearing distance
                        float threshold = (sourceLocal == SRC_SAY_LOCAL) ? g_SayDistance : g_YellDistance;
                        bool hasRealPlayerNearby = false;
                        
                        if (candidate->IsInWorld() && threshold > 0.0f)
                        {
                            for (auto const& nearbyPlayerItr : allPlayers)
                            {
                                Player* nearbyPlayer = nearbyPlayerItr.second;
                                if (nearbyPlayer && nearbyPlayer->IsInWorld())
                                {
                                    PlayerbotAI* nearbyAI = PlayerbotsMgr::instance().GetPlayerbotAI(nearbyPlayer);
                                    if (!nearbyAI || !nearbyAI->IsBotAI())
                                    {
                                        if (candidate->GetDistance(nearbyPlayer) <= threshold)
                                        {
                                            hasRealPlayerNearby = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (!hasRealPlayerNearby)
                            continue; // Skip bot - no real player can hear Say/Yell
                    }
                    
                    eligibleBots.push_back(candidate);
                }
            }
        }
    }
    
    std::vector<Player*> candidateBots;
    int notEligibleCount = 0;
    for (Player* bot : eligibleBots)
    {
        if (!bot)
        {
            continue;
        }
        
        // For channel messages, bots in eligibleBots have already passed STRICT channel checks
        // Only run additional eligibility checks for non-channel sources
        // EXCEPTION: If channel is nullptr but sourceLocal is a channel type (like GENERAL), 
        // treat it as a channel message (happens with bot-initiated messages)
        bool isChannelSource = (sourceLocal == SRC_GENERAL_LOCAL);
        
        if (channel != nullptr || isChannelSource)
        {
            // Channel bots have already been verified to be in EXACT same channel instance
            // OR this is a channel-type source (General) even without channel object
            candidateBots.push_back(bot);
        }
        else
        {
            // For non-channel sources (Say/Yell/Guild/Party/Whisper), run the full eligibility check
            if (IsBotEligibleForChatChannelLocal(bot, player, sourceLocal, channel, receiver))
                candidateBots.push_back(bot);
            else
                notEligibleCount++;
        }
    }
    
    if (g_DebugEnabled && notEligibleCount > 0)
    {
        LOG_INFO("server.loading", "[Ollama Chat] {} bots not eligible for {} (distance/guild/party checks failed)", 
                notEligibleCount, ChatChannelSourceLocalStr[sourceLocal]);
    }
    
    // Determine reply chance based on channel type
    uint32_t chance;
    if (sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL)
    {
        // Say/Yell channel type
        chance = senderIsBot ? g_BotReplyChance_Say : g_PlayerReplyChance_Say;
    }
    else if (sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL)
    {
        // Party/Raid channel type
        chance = senderIsBot ? g_BotReplyChance_Party : g_PlayerReplyChance_Party;
    }
    else if (sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL)
    {
        // Guild/Officer channel type
        chance = senderIsBot ? g_BotReplyChance_Guild : g_PlayerReplyChance_Guild;
    }
    else if (sourceLocal == SRC_GENERAL_LOCAL)
    {
        // General/Trade/Custom channel type
        chance = senderIsBot ? g_BotReplyChance_Channel : g_PlayerReplyChance_Channel;
    }
    else
    {
        // Default fallback (whispers, etc.) - use Say chances
        chance = senderIsBot ? g_BotReplyChance_Say : g_PlayerReplyChance_Say;
    }
    
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Sender: {} ({}), Channel: {}, Reply Chance: {}%, Candidate Bots: {}",
                player->GetName(), senderIsBot ? "BOT" : "PLAYER", ChatChannelSourceLocalStr[sourceLocal], chance, candidateBots.size());
    }
    
    std::vector<Player*> finalCandidates;
    
    // For whispers, handle directly - there should only be one receiver bot
    if (sourceLocal == SRC_WHISPER_LOCAL)
    {
        if (!candidateBots.empty())
        {
            Player* whisperBot = candidateBots[0]; // Should only be one bot for whispers
            if (!(g_DisableRepliesInCombat && whisperBot->IsInCombat()))
            {
                finalCandidates.push_back(whisperBot);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Whisper: Bot {} selected to respond", whisperBot->GetName());
                }
            }
        }
    }
    else
    {
        // Handle non-whisper chats with normal multi-bot logic
        std::vector<std::pair<size_t, Player*>> mentionedBots;

        // Helper to convert string to lowercase safely
        auto toLowerStr = [](const std::string& str) -> std::string {
            std::string result = str;
            for (char& c : result)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
            return result;
        };

        // Helper to check if a bot name is mentioned as a complete word
        auto isBotNameMentioned = [&trimmedMsg, &toLowerStr](const std::string& botName) -> size_t {
            std::string lowerMsg = toLowerStr(trimmedMsg);
            std::string lowerBotName = toLowerStr(botName);
            
            size_t pos = 0;
            while ((pos = lowerMsg.find(lowerBotName, pos)) != std::string::npos)
            {
                // Check if it's a word boundary before the name
                bool validStart = (pos == 0 || !std::isalnum(static_cast<unsigned char>(lowerMsg[pos - 1])));
                // Check if it's a word boundary after the name
                size_t endPos = pos + lowerBotName.length();
                bool validEnd = (endPos >= lowerMsg.length() || !std::isalnum(static_cast<unsigned char>(lowerMsg[endPos])));
                
                if (validStart && validEnd)
                {
                    return pos; // Found a valid word-boundary match
                }
                pos++; // Continue searching
            }
            return std::string::npos;
        };

        for (Player* bot : candidateBots)
        {
            if (!bot)
            {
                continue;
            }
            if (g_DisableRepliesInCombat && bot->IsInCombat())
            {
                continue;
            }
            
            size_t pos = isBotNameMentioned(bot->GetName());
            if (pos != std::string::npos)
            {
                mentionedBots.emplace_back(pos, bot);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} mentioned at position {} in message", bot->GetName(), pos);
                }
            }
        }

        if (!mentionedBots.empty())
        {
            // Sort by position to get the first mentioned bot
            std::sort(mentionedBots.begin(), mentionedBots.end(),
                      [](const std::pair<size_t, Player*> &a, const std::pair<size_t, Player*> &b) { return a.first < b.first; });
            Player* chosen = mentionedBots.front().second;
            if (!(g_DisableRepliesInCombat && chosen->IsInCombat()))
            {
                finalCandidates.push_back(chosen);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} selected (mentioned first at position {})", 
                            chosen->GetName(), mentionedBots.front().first);
                }
            }
        }
        else if (!senderIsBot && sourceLocal == SRC_GENERAL_LOCAL)
        {
            Player* nearest = nullptr;
            float nearestDist = 1e9f;
            for (Player* bot : candidateBots)
            {
                if (g_DisableRepliesInCombat && bot->IsInCombat())
                    continue;
                float dist = player->GetDistance(bot);
                if (dist < nearestDist)
                {
                    nearestDist = dist;
                    nearest = bot;
                }
            }
            if (nearest)
            {
                finalCandidates.push_back(nearest);
                if (candidateBots.size() > 1)
                {
                    for (uint32_t i = 0; i < 8; ++i)
                    {
                        Player* other = candidateBots[urand(0, static_cast<uint32_t>(candidateBots.size()) - 1)];
                        if (other != nearest && !(g_DisableRepliesInCombat && other->IsInCombat()))
                        {
                            finalCandidates.push_back(other);
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            for (Player* bot : candidateBots)
            {
                if (g_DisableRepliesInCombat && bot->IsInCombat())
                {
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipped - in combat", bot->GetName());
                    }
                    continue;
                }
                uint32_t roll = urand(0, 99);
                if (roll < chance)
                {
                    finalCandidates.push_back(bot);
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} PASSED chance roll ({} < {}%)", bot->GetName(), roll, chance);
                    }
                }
                else if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} FAILED chance roll ({} >= {}%)", bot->GetName(), roll, chance);
                }
            }
        }
    }

    
    if (finalCandidates.empty())
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] *** NO BOTS RESPONDING *** to {} from {} in {} channel. "
                    "Eligible: {}, Candidates: {}, Final: 0, Chance: {}%",
                    senderIsBot ? "BOT" : "PLAYER", player->GetName(), ChatChannelSourceLocalStr[sourceLocal],
                    eligibleBots.size(), candidateBots.size(), chance);
            LOG_INFO("server.loading", "[Ollama Chat] No eligible bots found to respond to message '{}'. "
                    "Source: {}, Eligible bots: {}, Candidate bots: {}, Combat disabled: {}",
                    msg, ChatChannelSourceLocalStr[sourceLocal], eligibleBots.size(), 
                    candidateBots.size(), g_DisableRepliesInCombat);
        }
        return;
    }
    
    uint32_t maxPick = g_MaxBotsToPick;
    if (senderIsBot && sourceLocal == SRC_GENERAL_LOCAL)
        maxPick = 1;
    if (!senderIsBot && sourceLocal == SRC_GENERAL_LOCAL)
        maxPick = kGeneralMaxBotsPerHumanMessage;

    if (finalCandidates.size() > maxPick)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(finalCandidates.begin(), finalCandidates.end(), g);
        uint32_t countToPick = urand(1, maxPick);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Limiting {} bots to {} (MaxBotsToPick)", finalCandidates.size(), countToPick);
        }
        finalCandidates.resize(countToPick);
    }
    
    if(g_DebugEnabled && !finalCandidates.empty())
    {
        std::string botNames;
        for (Player* bot : finalCandidates)
        {
            if (!botNames.empty()) botNames += ", ";
            botNames += bot->GetName();
        }
        LOG_INFO("server.loading", "[Ollama Chat] *** {} BOTS RESPONDING *** to {} from {} in {}: [{}]",
                finalCandidates.size(), senderIsBot ? "BOT" : "PLAYER", player->GetName(),
                ChatChannelSourceLocalStr[sourceLocal], botNames);
    }
    
    uint64_t senderGuid = player->GetGUID().GetRawValue();
    
    for (Player* bot : finalCandidates)
    {
        float distance = player->GetDistance(bot);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Bot {} (distance: {}) is set to respond.", bot->GetName(), distance);
        }
        if (bot == nullptr) {
            continue;
        }
        uint64_t botGuid = bot->GetGUID().GetRawValue();
        std::string msgCopy = msg;
        ChatIntent intent = DetectChatIntent(msgCopy, CollectReferenceNames(bot, candidateBots));

        std::thread([botGuid, senderGuid, msgCopy, sourceLocal, channelId = (channel ? channel->GetChannelId() : 0), channelName = (channel ? channel->GetName() : ""), intent]() {
            try {
                Player* botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                Player* senderPtr = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                if (!botPtr || !senderPtr)
                    return;

                PromptBundle promptBundle = BuildPlayerChatPrompt(botPtr, senderPtr, msgCopy, sourceLocal, intent);

                if (g_DebugEnabled && g_DebugShowFullPrompt)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Full prompt for bot {} <- {}: system=[{}] user=[{}]",
                        botPtr->GetName(), senderPtr->GetName(), promptBundle.system, promptBundle.user);
                }

                auto responseFuture = SubmitQuery(std::move(promptBundle));
                if (!responseFuture.valid())
                {
                    return;
                }
                std::string response = responseFuture.get();
                bool firstPassVerified = true;

                VerifyResult vr = VerifyReply(intent, msgCopy, response, sourceLocal);
                if (!vr.pass && kReplyVerificationMaxRetries > 0)
                {
                    firstPassVerified = false;
                    promptBundle = BuildPlayerChatPrompt(botPtr, senderPtr, msgCopy, sourceLocal, intent, vr.feedback);
                    responseFuture = SubmitQuery(std::move(promptBundle));
                    if (responseFuture.valid())
                        response = responseFuture.get();
                }

                botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                senderPtr = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                if (!botPtr)
                {
                    LOG_WARN("server.loading", "[Ollama Chat] Bot {} logged off before reply", botGuid);
                    return;
                }
                if (!senderPtr)
                {
                    LOG_WARN("server.loading", "[Ollama Chat] Sender {} logged off before reply", senderGuid);
                    return;
                }
                PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(senderPtr);
                bool const senderIsBot = senderAI && senderAI->IsBotAI();
                if (response.empty())
                {
                    LOG_WARN("server.loading", "[Ollama Chat] {} skipped reply — empty LLM response", botPtr->GetName());
                    return;
                }
                PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                if (!botAI)
                {
                    if(g_DebugEnabled)
                    {
                        LOG_ERROR("server.loading", "[Ollama Chat] No PlayerbotAI found for bot {}", botPtr->GetName());
                    }
                    return;
                }
                
                // Simulate typing delay if enabled (skip for General zone chat)
                if (g_EnableTypingSimulation && sourceLocal != SRC_GENERAL_LOCAL)
                {
                    uint32_t delay = g_TypingSimulationBaseDelay + (response.length() * g_TypingSimulationDelayPerChar);
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[OllamaChat] Bot {} simulating typing delay: {}ms for {} characters", 
                                 botPtr->GetName(), delay, response.length());
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                    
                    // Reacquire pointers after delay
                    botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                    if (!botPtr) return;
                    botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                    if (!botAI) return;
                    senderPtr = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                    if (!senderPtr) return;
                }
                
                // Route the response.
                if (channelId != 0 && !channelName.empty())
                {
                    // For channels, get the channel instance for the bot's team
                    ChannelMgr* cMgr = ChannelMgr::forTeam(botPtr->GetTeamId());
                    if (cMgr)
                    {
                        Channel* targetChannel = cMgr->GetChannel(channelName, botPtr, false);
                        if (targetChannel)
                        {
                            EnsureBotInChannel(botPtr, targetChannel);
                            if (botPtr->IsInChannel(targetChannel))
                            {
                                if(g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} sending to channel '{}' (ID: {})",
                                            botPtr->GetName(), channelName, targetChannel->GetChannelId());
                                }
                                if (sourceLocal == SRC_GENERAL_LOCAL)
                                {
                                    if (!TrySendGeneralChat(botPtr, response, msgCopy, targetChannel))
                                        return;
                                }
                                else
                                {
                                    targetChannel->Say(botPtr->GetGUID(), response, LANG_UNIVERSAL);
                                    ProcessBotChatMessage(botPtr, response, sourceLocal, targetChannel);
                                }
                                if(g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} responded in channel {}: {}",
                                            botPtr->GetName(), channelName, response);
                                }
                            }
                            else
                            {
                                LOG_WARN("server.loading", "[Ollama Chat] {} not in channel '{}' — reply skipped",
                                    botPtr->GetName(), channelName);
                            }
                        }
                        else
                        {
                            LOG_WARN("server.loading", "[Ollama Chat] {} channel '{}' not found — reply skipped",
                                botPtr->GetName(), channelName);
                        }
                    }
                }
                else
                {
                    switch (sourceLocal)
                    {
                        case SRC_GUILD_LOCAL: 
                            botAI->SayToGuild(response);
                            ProcessBotChatMessage(botPtr, response, SRC_GUILD_LOCAL, nullptr);
                            break;
                        case SRC_OFFICER_LOCAL: 
                            botAI->SayToGuild(response);
                            ProcessBotChatMessage(botPtr, response, SRC_OFFICER_LOCAL, nullptr);
                            break;
                        case SRC_PARTY_LOCAL: 
                            botAI->SayToParty(response);
                            ProcessBotChatMessage(botPtr, response, SRC_PARTY_LOCAL, nullptr);
                            break;
                        case SRC_RAID_LOCAL:  
                            botAI->SayToRaid(response);
                            ProcessBotChatMessage(botPtr, response, SRC_RAID_LOCAL, nullptr);
                            break;
                        case SRC_SAY_LOCAL:
                            // Only send Say if someone (real player or bot) is within say distance
                            {
                                bool someoneCanHear = false;
                                if (botPtr->IsInWorld())
                                {
                                    for (auto const& pair : ObjectAccessor::GetPlayers())
                                    {
                                        Player* nearbyPlayer = pair.second;
                                        if (nearbyPlayer && nearbyPlayer != botPtr && nearbyPlayer->IsInWorld())
                                        {
                                            if (botPtr->GetDistance(nearbyPlayer) <= g_SayDistance)
                                            {
                                                someoneCanHear = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (someoneCanHear)
                                {
                                    botAI->Say(response);
                                    ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                                }
                                else if (g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping Say reply - no one within {} yards to hear it", 
                                            botPtr->GetName(), g_SayDistance);
                                }
                            }
                            break;
                        case SRC_YELL_LOCAL:
                            // Only send Yell if someone is within yell distance
                            {
                                bool someoneCanHear = false;
                                if (botPtr->IsInWorld())
                                {
                                    for (auto const& pair : ObjectAccessor::GetPlayers())
                                    {
                                        Player* nearbyPlayer = pair.second;
                                        if (nearbyPlayer && nearbyPlayer != botPtr && nearbyPlayer->IsInWorld())
                                        {
                                            if (botPtr->GetDistance(nearbyPlayer) <= g_YellDistance)
                                            {
                                                someoneCanHear = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (someoneCanHear)
                                {
                                    botAI->Yell(response);
                                    ProcessBotChatMessage(botPtr, response, SRC_YELL_LOCAL, nullptr);
                                }
                                else if (g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping Yell reply - no one within {} yards to hear it", 
                                            botPtr->GetName(), g_YellDistance);
                                }
                            }
                            break;
                        case SRC_WHISPER_LOCAL:
                            // For whispers, find the original sender and whisper back
                            {
                                Player* originalSender = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                                if (originalSender)
                                {
                                    if(g_DebugEnabled)
                                    {
                                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} whispering response '{}' to {}", 
                                                botPtr->GetName(), response, originalSender->GetName());
                                    }
                                    botAI->Whisper(response, originalSender->GetName());
                                    // Don't trigger ProcessBotChatMessage for whispers - they're private
                                }
                                else if(g_DebugEnabled)
                                {
                                    LOG_ERROR("server.loading", "[Ollama Chat] Cannot whisper response - original sender not found for GUID {}", senderGuid);
                                }
                            }
                            break;
                        default:              
                            botAI->Say(response);
                            ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                            break;
                    }
                }
                
                // Update sentiment based on the player's message
                UpdateBotPlayerSentiment(botPtr, senderPtr, msgCopy);

                AppendBotConversation(botGuid, senderGuid, msgCopy, response, false, sourceLocal, senderIsBot, firstPassVerified);
                if (botPtr->IsInWorld() && senderPtr->IsInWorld())
                {
                    float respDistance = senderPtr->GetDistance(botPtr);
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} (distance: {}) responded: {}", botPtr->GetName(), respDistance, response);
                    }
                }
                else
                {
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} responded: {} (distance not calculated - players not in world)", botPtr->GetName(), response);
                    }
                }
            }
            catch (const std::exception& ex)
            {
                if(g_DebugEnabled)
                {
                    LOG_ERROR("server.loading", "[Ollama Chat] Exception in bot response thread: {}", ex.what());
                }
            }
        }).detach();

    }
}

static bool IsBotEligibleForChatChannelLocal(Player* bot, Player* player, ChatChannelSourceLocal source, Channel* channel, Player* receiver)
{
    if (!bot || !player || bot == player)
    {
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] IsBotEligible: FAILED basic check - bot={}, player={}, same={}", 
                    (void*)bot, (void*)player, (bot == player));
        return false;
    }
    if (!PlayerbotsMgr::instance().GetPlayerbotAI(bot))
    {
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] IsBotEligible: Bot {} FAILED - no PlayerbotAI", bot->GetName());
        return false;
    }
        
    // For whispers, only the specific receiver should respond
    if (source == SRC_WHISPER_LOCAL)
    {
        // Don't allow bot-to-bot whisper responses
        PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
        if (senderAI && senderAI->IsBotAI())
        {
            return false;
        }
        
        return (receiver && bot == receiver);
    }
    
    // Check team compatibility for non-proximity chats (except channels which can be cross-faction)
    // Say and Yell are proximity-based and don't require same faction
    bool isProximityChatSource = (source == SRC_SAY_LOCAL || source == SRC_YELL_LOCAL);
    if (!channel && !isProximityChatSource && bot->GetTeamId() != player->GetTeamId())
        return false;
    
    // For channels, check if bot is in the specific channel instance
    if (channel)
    {
        // Verify the channel is valid before proceeding
        if (!channel)
        {
            if(g_DebugEnabled)
            {
                LOG_ERROR("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Channel is null");
            }
            return false;
        }
            
        // ONLY use exact channel instance check - NO Player::IsInChannel() anymore
        ChannelMgr* candidateCMgr = ChannelMgr::forTeam(bot->GetTeamId());
        if (!candidateCMgr)
            return false;
            
        Channel* candidateChannel = candidateCMgr->GetChannel(channel->GetName(), bot);
        // Verify both channels are valid and are the exact same instance
        if (!candidateChannel || candidateChannel != channel)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Bot {} not in same channel instance '{}' - Bot team: {}, Channel ptr: {} vs {}", 
                        bot->GetName(), channel->GetName(), (int)bot->GetTeamId(),
                        (void*)candidateChannel, (void*)channel);
            }
            return false;
        }
        
        // Additional team check for cross-faction channels - only allow same faction unless it's a global channel
        if (bot->GetTeamId() != player->GetTeamId())
        {
            // Allow cross-faction only for specific global channels
            bool isGlobalChannel = (channel->GetName().find("World") != std::string::npos || 
                                   channel->GetName().find("LookingForGroup") != std::string::npos);
            if (!isGlobalChannel)
            {
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Bot {} different faction from player - Bot: {}, Player: {}, Channel: '{}'", bot->GetName(), (int)bot->GetTeamId(), (int)player->GetTeamId(), channel->GetName());
                }
                return false;
            }
        }
    }
    
    bool isInParty = (player->GetGroup() && bot->GetGroup() && (player->GetGroup() == bot->GetGroup()));
    float threshold = 0.0f;
    
    switch (source)
    {
        case SRC_SAY_LOCAL:    
            threshold = g_SayDistance;
            if (threshold > 0.0f)
            {
                if (!bot->IsInWorld() || !player->IsInWorld())
                    return false;
                    
                float distance = bot->GetDistance(player);
                return distance <= threshold;
            }
            return false;
            
        case SRC_YELL_LOCAL:   
            threshold = g_YellDistance;
            return (threshold > 0.0f && player->GetDistance(bot) <= threshold);
            
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            return (player->GetGuild() && bot->GetGuildId() == player->GetGuildId());
            
        case SRC_PARTY_LOCAL:
        case SRC_RAID_LOCAL:
            return isInParty;
            
        case SRC_WHISPER_LOCAL:
            // For whispers, the bot should only respond if it's the specific receiver
            return (receiver && bot == receiver);
            
        case SRC_GENERAL_LOCAL:
            // For channels like General, Trade, etc., no distance check - only channel membership matters
            // Channel membership was already checked above
            return true;
            
        default:
            return false;
    }
}

