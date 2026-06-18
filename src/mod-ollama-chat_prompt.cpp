#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include "Player.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "AiFactory.h"
#include "Guild.h"
#include "SharedDefines.h"
#include "Util.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace
{

std::string g_SystemPrompt;
std::string g_ChatTask;
std::string g_RandomTask;
std::string g_EventTask;
std::string g_SentimentTask;
std::string g_HedgeSection;
std::string g_VagueKnowledgeSection;
std::string g_FactualHintSection;
std::string g_MemoryMaintenanceTask;
bool g_PromptFilesLoaded = false;
std::string g_ChannelGeneral;
std::string g_ChannelGeneralRandom;

constexpr char const* kPromptFallbackPaths[] = {
    "/azerothcore/modules/mod-ollama-chat/data/prompts/",
    "../../../modules/mod-ollama-chat/data/prompts/",
    "modules/mod-ollama-chat/data/prompts/",
    "data/prompts/",
};

bool PromptDirHasSystemTxt(std::string const& path)
{
    try
    {
        std::string p = path;
        if (!p.empty() && p.back() != '/')
            p += '/';
        return fs::exists(p + "system.txt");
    }
    catch (std::exception const&) {}
    return false;
}

std::string ResolvePromptDataPath(std::string const& configured)
{
    if (!configured.empty() && PromptDirHasSystemTxt(configured))
        return configured;
    for (char const* fallback : kPromptFallbackPaths)
    {
        if (PromptDirHasSystemTxt(fallback))
            return fallback;
    }
    return configured.empty() ? kPromptFallbackPaths[0] : configured;
}

std::string Trim(std::string s)
{
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

std::string LoadTextFile(std::string const& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        return "";
    std::ostringstream ss;
    ss << file.rdbuf();
    return Trim(ss.str());
}

void AppendSection(std::ostringstream& out, std::string const& title, std::string const& body)
{
    if (body.empty())
        return;
    out << "## " << title << "\n" << body << "\n\n";
}

std::string const& HomeZoneForRace(uint8_t race)
{
    static std::string const mulgore = "Mulgore";
    static std::string const durotar = "Durotar";
    static std::string const teldrassil = "Teldrassil";
    static std::string const dunMorogh = "Dun Morogh";
    static std::string const tirisfal = "Tirisfal Glades";
    static std::string const eversong = "Eversong Woods";
    static std::string const azuremyst = "Azuremyst Isle";
    static std::string const empty;
    switch (race)
    {
        case RACE_TAUREN: return mulgore;
        case RACE_ORC:
        case RACE_TROLL: return durotar;
        case RACE_NIGHTELF: return teldrassil;
        case RACE_DWARF:
        case RACE_GNOME: return dunMorogh;
        case RACE_UNDEAD_PLAYER: return tirisfal;
        case RACE_BLOODELF: return eversong;
        case RACE_DRAENEI: return azuremyst;
        default: return empty;
    }
}

int PersonalityKnowledgeModifier(std::string const& key)
{
    if (key == "FOOL" || key == "YOUNG_APPRENTICE")
        return -15;
    if (key == "SCHOLAR" || key == "MENTOR" || key == "ANCIENT_WISE_ONE")
        return 15;
    if (key == "LONE_WOLF")
        return -5;
    if (key == "CASUAL" || key == "ROLEPLAYER")
        return 5;
    return 0;
}

bool ZoneTagMatches(std::vector<RAGResult> const& results, std::string const& zoneLower)
{
    if (zoneLower.empty())
        return false;
    for (auto const& r : results)
    {
        if (!r.entry)
            continue;
        for (auto const& tag : r.entry->tags)
        {
            if (tag.find("zone:") == 0)
            {
                std::string const& z = tag.substr(5);
                if (zoneLower.find(z) != std::string::npos || z.find(zoneLower) != std::string::npos)
                    return true;
            }
            if (zoneLower.find(tag) != std::string::npos)
                return true;
        }
    }
    return false;
}

static bool RagEntryTooHighForLevel(RAGEntry const* entry, uint32_t level)
{
    if (!entry)
        return false;
    for (std::string const& tag : entry->tags)
    {
        if (tag.rfind("level:", 0) != 0)
            continue;
        size_t dash = tag.find('-', 6);
        if (dash == std::string::npos)
            continue;
        char* end = nullptr;
        long maxLevel = std::strtol(tag.c_str() + dash + 1, &end, 10);
        if (end != tag.c_str() + dash + 1 && maxLevel > static_cast<long>(level) + 10)
            return true;
    }
    return false;
}

static void FilterRagByLevel(std::vector<RAGResult>& results, uint32_t level)
{
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [level](RAGResult const& r) { return RagEntryTooHighForLevel(r.entry, level); }),
        results.end());
}

static std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

static bool RagClassEntryMatches(RAGEntry const* entry, std::string const& className)
{
    if (!entry || className.empty())
        return false;
    std::string lower = ToLower(className);
    if (ToLower(entry->id).find(lower) != std::string::npos)
        return true;
    for (std::string const& kw : entry->keywords)
    {
        std::string kwLower = ToLower(kw);
        if (kwLower == lower || kwLower.find(lower) != std::string::npos)
            return true;
    }
    return false;
}

static void FilterRagForCasualChatter(std::vector<RAGResult>& results)
{
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [](RAGResult const& r) {
                if (!r.entry)
                    return true;
                for (std::string const& tag : r.entry->tags)
                {
                    if (tag == "economy" || tag == "gold" || tag == "professions" ||
                        tag == "auction" || tag == "general")
                        return true;
                }
                return false;
            }),
        results.end());
}

static void FilterRagToZoneOnly(std::vector<RAGResult>& results, std::string const& zoneLower)
{
    if (zoneLower.empty())
    {
        results.clear();
        return;
    }
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [&](RAGResult const& r) { return !ZoneTagMatches({ r }, zoneLower); }),
        results.end());
}

static void FilterRagByClass(std::vector<RAGResult>& results, std::string const& botClass, std::string const& playerClass)
{
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [&](RAGResult const& r) {
                if (!r.entry)
                    return true;
                auto const& tags = r.entry->tags;
                if (std::find(tags.begin(), tags.end(), "class") == tags.end())
                    return false;
                return !RagClassEntryMatches(r.entry, botClass) && !RagClassEntryMatches(r.entry, playerClass);
            }),
        results.end());
}

static void AppendSpecIfDistinct(std::ostringstream& out, std::string const& role, std::string const& cls,
    char const* prefix = " (", char const* suffix = ")")
{
    if (!role.empty() && role != cls)
        out << prefix << role << suffix;
}

static char const* ChannelLabel(ChatChannelSourceLocal channel)
{
    switch (channel)
    {
        case SRC_SAY_LOCAL: return "Say";
        case SRC_PARTY_LOCAL: return "Party";
        case SRC_RAID_LOCAL: return "Raid";
        case SRC_GUILD_LOCAL: return "Guild";
        case SRC_OFFICER_LOCAL: return "Officer";
        case SRC_YELL_LOCAL: return "Yell";
        case SRC_WHISPER_LOCAL: return "Whisper";
        case SRC_GENERAL_LOCAL: return "General";
        default: return "Chat";
    }
}

static std::string BuildClassSection(BotContext const& ctx, Player* /*playerOrNull*/)
{
    std::ostringstream ss;
    ss << "You: " << ctx.botRace << " " << ctx.botClass;
    AppendSpecIfDistinct(ss, ctx.botRole, ctx.botClass);
    ss << ". Only mention abilities your class can use.";
    if (!ctx.playerName.empty())
    {
        ss << "\nPlayer " << ctx.playerName << ": " << ctx.playerClass;
        AppendSpecIfDistinct(ss, ctx.playerRole, ctx.playerClass);
        ss << ". When advising them, only suggest their class's abilities.";
    }
    return ss.str();
}

static std::string BuildSituationSection(BotContext const& ctx, Player* bot, Player* playerOrNull, ChatChannelSourceLocal channel)
{
    std::ostringstream ss;
    ss << "Channel: " << ChannelLabel(channel);
    if (channel == SRC_GENERAL_LOCAL)
        ss << " (/1). Strangers in zone. One blunt line. No RP speech, no guides, no poetry.";
    if (bot)
    {
        ss << "\nCombat: you " << (bot->IsInCombat() ? "in combat" : "not in combat");
        if (playerOrNull)
            ss << " | " << ctx.playerName << " " << (playerOrNull->IsInCombat() ? "in combat" : "not in combat");
    }
    if (ctx.groupCtx.memberCount == 0)
        ss << "\nAudience: solo, no group";
    else if (ctx.groupCtx.memberCount == 2)
        ss << "\nAudience: only you and one other (" << ctx.groupCtx.playerCount << " player"
           << (ctx.groupCtx.playerCount == 1 ? "" : "s") << ", " << ctx.groupCtx.botCount << " bot"
           << (ctx.groupCtx.botCount == 1 ? "" : "s") << ")";
    else
        ss << "\nAudience: party of " << ctx.groupCtx.memberCount << " (" << ctx.groupCtx.playerCount
           << " players, " << ctx.groupCtx.botCount << " bots)";
    return ss.str();
}

static std::string MergeSystemPrompts()
{
    if (!g_SystemPromptOverride.empty())
        return g_SystemPromptOverride;
    std::ostringstream ss;
    if (!g_OllamaSystemPrompt.empty())
        ss << g_OllamaSystemPrompt;
    if (!g_SystemPrompt.empty())
    {
        if (ss.tellp() > 0)
            ss << "\n";
        ss << g_SystemPrompt;
    }
    return ss.str();
}
} // namespace

std::string RandomIntentTaskLine(RandomIntent intent, ChatChannelSourceLocal channel)
{
    if (channel == SRC_GENERAL_LOCAL)
        return "Say one casual zone-chat line.";

    switch (intent)
    {
        case RandomIntent::ObserveEnvironment:
            return "Comment briefly on what you notice around you.";
        case RandomIntent::SmallTalk:
            return "Make a short casual remark about the zone or what you are doing.";
        case RandomIntent::AskGroup:
            return "Ask one short question to nearby players about the area or grouping.";
        case RandomIntent::ObserveZone:
        default:
            return "Make a short observation about this zone or your current activity.";
    }
}

std::string PromptBundle::CombinedForLegacyGenerate() const
{
    if (!system.empty() && !user.empty())
        return user;
    if (!user.empty())
        return user;
    return system;
}

void OllamaPromptComposer::LoadPromptFiles()
{
    std::string base = ResolvePromptDataPath(g_PromptDataPath);
    if (base != g_PromptDataPath)
        g_PromptDataPath = base;
    if (base.empty())
        base = kPromptFallbackPaths[0];

    auto load = [&](std::string const& rel, std::string& dest, std::string const& fallback) {
        std::string path = base + rel;
        dest = LoadTextFile(path);
        if (dest.empty())
            dest = fallback;
    };

    load("system.txt", g_SystemPrompt,
         "You are a Wrath-era WoW player. Reply in under 15 words in character.");
    load("chat_task.txt", g_ChatTask, "Reply to the player's message only.");
    load("random_task.txt", g_RandomTask, "Say one short casual line.");
    load("event_task.txt", g_EventTask, "React briefly to the event.");
    load("sentiment_task.txt", g_SentimentTask,
         "Respond with only POSITIVE, NEGATIVE, or NEUTRAL.");
    load("sections/hedge.txt", g_HedgeSection,
         "You are not sure about this. Say so in character.");
    load("sections/vague_knowledge.txt", g_VagueKnowledgeSection,
         "You vaguely remember:");
    load("sections/factual_hint.txt", g_FactualHintSection,
         "Answer the factual question directly or admit uncertainty.");
    load("memory_flush.txt", g_MemoryMaintenanceTask,
         "Update player memory from chat. Reply only.\n[FACTS]\nOne fact per line.\n[NOTES]\nPlayer preferences only.");
    load("channels/general.txt", g_ChannelGeneral, "");
    load("channels/general_random.txt", g_ChannelGeneralRandom, "");

    if (!g_SystemPromptOverride.empty())
        g_SystemPrompt = g_SystemPromptOverride;

    g_PromptFilesLoaded = true;
}

std::string GetMemoryMaintenanceSystemPrompt()
{
    if (!g_PromptFilesLoaded)
        OllamaPromptComposer::LoadPromptFiles();
    return g_MemoryMaintenanceTask;
}

BotContext OllamaPromptComposer::GatherBotContext(Player* bot, Player* playerOrNull)
{
    BotContext ctx;
    if (!bot)
        return ctx;

    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!botAI || !botAI->GetChatHelper())
        return ctx;

    ctx.personalityKey = GetBotPersonality(bot);
    ctx.personalityLine = GetPersonalityPromptAddition(ctx.personalityKey);
    ctx.botName = bot->GetName();
    ctx.botLevel = bot->GetLevel();
    ctx.botClass = botAI->GetChatHelper()->FormatClass(bot->getClass());
    ctx.botRace = botAI->GetChatHelper()->FormatRace(bot->getRace());
    ctx.botRole = ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot));
    ctx.botGender = bot->getGender() == GENDER_MALE ? "Male" : "Female";
    ctx.botFaction = bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde";
    ctx.botGuild = bot->GetGuild() ? bot->GetGuild()->GetName() : "No Guild";
    ctx.groupCtx = BuildGroupContext(bot);
    ctx.botInCombat = bot->IsInCombat();
    ctx.botRaceId = bot->getRace();

    AreaTableEntry const* area = botAI->GetCurrentArea();
    AreaTableEntry const* zone = botAI->GetCurrentZone();
    ctx.botArea = area ? botAI->GetLocalizedAreaName(area) : "Unknown";
    ctx.botZone = zone ? botAI->GetLocalizedAreaName(zone) : "Unknown";
    ctx.botMap = bot->GetMap() ? bot->GetMap()->GetMapName() : "Unknown";

    if (playerOrNull)
    {
        ctx.playerName = playerOrNull->GetName();
        ctx.playerLevel = playerOrNull->GetLevel();
        ctx.playerClass = botAI->GetChatHelper()->FormatClass(playerOrNull->getClass());
        ctx.playerRace = botAI->GetChatHelper()->FormatRace(playerOrNull->getRace());
        ctx.playerRole = ChatHelper::FormatClass(playerOrNull, AiFactory::GetPlayerSpecTab(playerOrNull));
        if (playerOrNull->IsInWorld() && bot->IsInWorld())
            ctx.playerDistance = bot->GetDistance(playerOrNull);
    }

    return ctx;
}

bool OllamaPromptComposer::IsFactualQuestion(std::string const& message)
{
    if (IsSocialPresenceQuestion(message))
        return false;
    std::string lower = message;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    static char const* markers[] = {
        "where is", "where's", "where are", "how do i", "how to", "which way",
        "location of", "find ", "coords", "coordinate", "what is the", "who is the"
    };
    for (char const* m : markers)
    {
        if (lower.find(m) != std::string::npos)
            return true;
    }
    return false;
}

ChatIntent DetectChatIntent(std::string const& message, std::vector<std::string> const& referenceNames);

bool OllamaPromptComposer::IsSocialPresenceQuestion(std::string const& message)
{
    return DetectChatIntent(message, {}).presenceQuestion;
}

namespace
{
bool ContainsWholeWord(std::string const& haystackLower, std::string const& wordLower)
{
    if (wordLower.empty())
        return false;
    size_t pos = 0;
    while ((pos = haystackLower.find(wordLower, pos)) != std::string::npos)
    {
        bool leftOk = pos == 0 || !std::isalnum(static_cast<unsigned char>(haystackLower[pos - 1]));
        size_t end = pos + wordLower.size();
        bool rightOk = end >= haystackLower.size() || !std::isalnum(static_cast<unsigned char>(haystackLower[end]));
        if (leftOk && rightOk)
            return true;
        pos += wordLower.size();
    }
    return false;
}

bool ContainsMarker(std::string const& lower, char const* const* markers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (lower.find(markers[i]) != std::string::npos)
            return true;
    }
    return false;
}
} // namespace

ChatIntent DetectChatIntent(std::string const& message, std::vector<std::string> const& referenceNames)
{
    ChatIntent intent;
    std::string lower = message;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    static char const* kPresence[] = { "anyone in", "anyone here", "anybody around", "anyone at", "anybody here" };
    static char const* kParty[] = { "party", "group up", "join me", "invite", "lfm", "want to group", "group?" };
    static char const* kQuestionStart[] = { "who ", "what ", "when ", "where ", "how ", "why " };

    intent.presenceQuestion = ContainsMarker(lower, kPresence, sizeof(kPresence) / sizeof(kPresence[0]));
    intent.partyInvite = ContainsMarker(lower, kParty, sizeof(kParty) / sizeof(kParty[0]));
    intent.directQuestion = !message.empty() && (message.back() == '?' ||
        ContainsMarker(lower, kQuestionStart, sizeof(kQuestionStart) / sizeof(kQuestionStart[0])));

    for (std::string const& name : referenceNames)
    {
        std::string nameLower = name;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
        if (ContainsWholeWord(lower, nameLower))
        {
            intent.botReference = true;
            intent.referencedName = nameLower;
            break;
        }
    }
    return intent;
}

std::string BuildIntentTaskLines(ChatIntent const& intent)
{
    std::string lines;
    if (intent.presenceQuestion)
        lines += "Answer presence only (yes/no/here). No quest tips unless asked.\n";
    if (intent.partyInvite)
        lines += "First few words: clear accept or decline, then optional banter.\n";
    if (intent.botReference && !intent.referencedName.empty())
        lines += "Acknowledge " + intent.referencedName + "; do not contradict unseen context.\n";
    if (intent.directQuestion)
        lines += "Answer the question directly in the first clause.\n";
    return lines;
}

std::string OllamaPromptComposer::BuildRagQuery(std::string const& message, std::string const& zone, std::string const& area)
{
    std::ostringstream q;
    q << message;
    if (!zone.empty() && zone != "Unknown")
        q << " " << zone;
    if (!area.empty() && area != "Unknown" && area != zone)
        q << " " << area;
    return q.str();
}

KnowledgeLevel OllamaPromptComposer::RollKnowledgeLevel(BotContext const& ctx, std::vector<RAGResult> const& results)
{
    if (results.empty())
        return KnowledgeLevel::Hedge;

    int chance = static_cast<int>(g_RAGKnowledgeChance);
    chance += PersonalityKnowledgeModifier(ctx.personalityKey);

    std::string zoneLower = ctx.botZone;
    std::transform(zoneLower.begin(), zoneLower.end(), zoneLower.begin(), ::tolower);

    if (ZoneTagMatches(results, zoneLower))
        chance += static_cast<int>(g_RAGKnowledgeZoneBonus);

    std::string home = HomeZoneForRace(ctx.botRaceId);
    std::string homeLower = home;
    std::transform(homeLower.begin(), homeLower.end(), homeLower.begin(), ::tolower);
    if (!homeLower.empty() && zoneLower.find(homeLower) != std::string::npos)
        chance += static_cast<int>(g_RAGKnowledgeHomeZoneBonus);

  // crude low-level penalty if bot is below 20 in high-level tagged content
    if (ctx.botLevel < 20)
        chance -= static_cast<int>(g_RAGKnowledgeLowLevelPenalty) / 2;

    chance = std::max(0, std::min(100, chance));

    int roll = urand(0, 99);
    if (roll >= chance)
        return KnowledgeLevel::Hedge;

    if (roll < chance / 4)
        return KnowledgeLevel::Vague;

    return KnowledgeLevel::Full;
}

std::string OllamaPromptComposer::BuildKnowledgeSection(KnowledgeLevel level, std::string const& ragBullets)
{
    switch (level)
    {
        case KnowledgeLevel::Hedge:
            return g_HedgeSection;
        case KnowledgeLevel::Vague:
            if (ragBullets.empty())
                return g_HedgeSection;
            return g_VagueKnowledgeSection + "\n" + ragBullets;
        case KnowledgeLevel::Full:
            if (ragBullets.empty())
                return g_HedgeSection;
            return ragBullets;
        case KnowledgeLevel::None:
        default:
            return "";
    }
}

PromptBundle OllamaPromptComposer::Build(PromptScenario scenario, BotContext const& ctx, ScenarioInput const& input)
{
    if (!g_PromptFilesLoaded)
        LoadPromptFiles();

    PromptBundle bundle;

    if (scenario == PromptScenario::SentimentAnalysis)
    {
        bundle.system = g_SentimentTask;
        bundle.user = input.playerMessage;
        return bundle;
    }

    if (scenario == PromptScenario::MemoryMaintenance)
    {
        bundle.system = GetMemoryMaintenanceSystemPrompt();
        std::ostringstream u;
        u << "Facts:\n" << (input.maintenanceFacts.empty() ? "(none)" : input.maintenanceFacts) << "\n\n";
        u << "Notes:\n" << (input.maintenanceNotes.empty() ? "(none)" : input.maintenanceNotes) << "\n\n";
        u << "Turns:\n" << input.maintenanceTurns;
        bundle.user = u.str();
        return bundle;
    }

    std::ostringstream user;
    std::ostringstream identity;
    identity << ctx.botName << " | L" << ctx.botLevel << " " << ctx.botRace << " " << ctx.botClass;
    AppendSpecIfDistinct(identity, ctx.botRole, ctx.botClass);
    identity << " | " << ctx.botArea << ", " << ctx.botZone << " | " << ctx.botFaction << " | "
             << ctx.groupCtx.statusLine;
    AppendSection(user, "Identity", identity.str());

    if (!input.contextSection.empty())
        AppendSection(user, "Context", input.contextSection);

    if (!ctx.personalityLine.empty())
    {
        bool const general = input.chatChannel == SRC_GENERAL_LOCAL;
        std::string persona = (general || ctx.personalityKey.empty())
            ? ctx.personalityLine
            : ctx.personalityKey + ": " + ctx.personalityLine;
        AppendSection(user, "Personality", persona);
    }

    if (!input.sentimentSection.empty())
        AppendSection(user, "Sentiment", input.sentimentSection);

    if (!input.memorySection.empty())
        AppendSection(user, "Memory", input.memorySection);

    if (!input.chatHistorySection.empty())
        AppendSection(user, "History", input.chatHistorySection);

    if (!input.recentGeneralSection.empty())
        AppendSection(user, "RecentGeneral", input.recentGeneralSection);

    if (!input.knowledgeSection.empty())
        AppendSection(user, "Knowledge", input.knowledgeSection);

    if (!input.environmentSection.empty())
        AppendSection(user, "Environment", input.environmentSection);

    if (!input.nearbySection.empty())
        AppendSection(user, "Nearby (visible, not party)", input.nearbySection);

    if (!input.eventType.empty())
    {
        std::ostringstream ev;
        if (!input.actorName.empty())
            ev << input.actorName << " ";
        ev << input.eventType << " " << input.eventDetail;
        AppendSection(user, "Event", ev.str());
    }

    if (input.factualQuestion)
        AppendSection(user, "Hint", g_FactualHintSection);

    std::ostringstream task;
    switch (scenario)
    {
        case PromptScenario::RandomChatter:
            task << g_RandomTask << "\n" << RandomIntentTaskLine(input.randomIntent, input.chatChannel);
            break;
        case PromptScenario::EventReaction:
            task << g_EventTask;
            if (input.chatChannel == SRC_GENERAL_LOCAL)
                task << "\nOne short reaction in zone chat, not a speech.";
            break;
        case PromptScenario::PlayerChat:
        default:
            if (!ctx.playerName.empty())
            {
                task << "[" << ChannelLabel(input.chatChannel) << "] " << ctx.playerName << " (L"
                     << ctx.playerLevel << " " << ctx.playerRace << " " << ctx.playerClass;
                AppendSpecIfDistinct(task, ctx.playerRole, ctx.playerClass, " ", "");
                task << ", " << static_cast<int>(ctx.playerDistance) << "yd): \"" << input.playerMessage << "\"\n";
            }
            else
            {
                task << "Message: \"" << input.playerMessage << "\"\n";
            }
            task << g_ChatTask;
            if (!input.intentTaskLines.empty())
                task << "\n" << input.intentTaskLines;
            if (!input.verificationFeedback.empty())
                task << "\nCorrection: " << input.verificationFeedback;
            if (ctx.groupCtx.memberCount > 0 &&
                (input.chatIntent.partyInvite || input.chatIntent.botReference))
                task << "\nIf invited to party or referenced by name, respond to that before jokes.";
            break;
    }
    AppendSection(user, "Task", task.str());

    bundle.user = user.str();
    return bundle;
}

std::string GetBotHistorySection(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage)
{
    std::string raw = GetBotHistoryPrompt(botGuid, playerGuid, playerMessage);
    if (raw.empty())
        return "";
    return raw;
}

std::string GetBotCompactNearbySection(Player* bot)
{
    if (!bot)
        return "";

    constexpr size_t kMaxBytes = 800;

    std::ostringstream ss;
    size_t byteCount = 0;

    auto appendLine = [&](std::string const& line) {
        if (byteCount >= kMaxBytes || line.empty())
            return false;
        if (byteCount > 0)
        {
            ss << "\n";
            ++byteCount;
        }
        ss << line;
        byteCount += line.size();
        return byteCount < kMaxBytes;
    };

    for (auto const& line : ChatHandler_GetVisibleLocations(bot, 40.0f))
    {
        if (!appendLine(line))
            break;
    }

    for (auto const& line : ChatHandler_GetVisiblePlayers(bot, 40.0f))
    {
        if (!appendLine(line))
            break;
    }

    return ss.str();
}

void EnrichPromptBundle(PromptBundle& bundle, Player* bot, BotContext const& ctx,
    Player* playerOrNull, ChatChannelSourceLocal channel, bool randomAmbient)
{
    if (!bot)
        return;

    if (!g_PromptFilesLoaded)
        OllamaPromptComposer::LoadPromptFiles();

    bundle.system = MergeSystemPrompts();
    if (channel == SRC_GENERAL_LOCAL)
    {
        if (!g_ChannelGeneral.empty())
        {
            if (!bundle.system.empty())
                bundle.system += "\n";
            bundle.system += g_ChannelGeneral;
        }
        if (randomAmbient && !g_ChannelGeneralRandom.empty())
            bundle.system += std::string("\n") + g_ChannelGeneralRandom;
    }

    std::ostringstream extra;
    AppendSection(extra, "Situation", BuildSituationSection(ctx, bot, playerOrNull, channel));
    AppendSection(extra, "Class", BuildClassSection(ctx, playerOrNull));
    if (ctx.groupCtx.memberCount > 0)
        AppendSection(extra, "Party", ctx.groupCtx.partySection);
    bundle.user = extra.str() + bundle.user;

    if (g_EnableChatBotSnapshotTemplate && channel != SRC_GENERAL_LOCAL)
        bundle.user += GenerateBotGameStateSnapshot(bot, ctx.groupCtx.memberCount > 0);
}

PromptBundle BuildPlayerChatPrompt(Player* bot, Player* player, std::string const& playerMessage,
    ChatChannelSourceLocal channel, ChatIntent const& intent, std::string const& verificationFeedback)
{
    BotContext ctx = OllamaPromptComposer::GatherBotContext(bot, player);
    ctx.personalityLine = GetPersonalityPromptForChannel(ctx.personalityKey, channel);

    if (!g_ChatPromptTemplate.empty())
    {
        PromptBundle bundle;
        bundle.user = GenerateBotPrompt(bot, playerMessage, player);
        EnrichPromptBundle(bundle, bot, ctx, player, channel);
        return bundle;
    }

    ScenarioInput input;
    input.playerMessage = playerMessage;
    input.factualQuestion = OllamaPromptComposer::IsFactualQuestion(playerMessage);
    input.chatChannel = channel;
    input.intentTaskLines = BuildIntentTaskLines(intent);
    input.chatIntent = intent;
    input.verificationFeedback = verificationFeedback;

    uint64_t botGuid = bot->GetGUID().GetRawValue();
    uint64_t playerGuid = player->GetGUID().GetRawValue();

    bool senderIsBot = false;
    if (player)
    {
        PlayerbotAI* pai = PlayerbotsMgr::instance().GetPlayerbotAI(player);
        senderIsBot = pai && pai->IsBotAI();
    }
    bool const skipContext = channel == SRC_GENERAL_LOCAL && senderIsBot;

    if (!skipContext)
        input.chatHistorySection = GetBotHistorySection(botGuid, playerGuid, playerMessage);

    if (g_EnableSentimentTracking && player)
        input.sentimentSection = GetSentimentPromptAddition(bot, player);

    if (g_EnableMemory && !skipContext)
        input.memorySection = GetMemoryPromptAddition(botGuid, playerGuid, channel, playerMessage, player->GetName());

    if (channel == SRC_GENERAL_LOCAL)
        input.recentGeneralSection = FormatRecentGeneralTranscript(bot->GetZoneId());

    if (g_EnableRAG && g_RAGSystem && !OllamaPromptComposer::IsSocialPresenceQuestion(playerMessage))
    {
        std::string ragQuery = OllamaPromptComposer::BuildRagQuery(playerMessage, ctx.botZone, ctx.botArea);
        auto results = g_RAGSystem->RetrieveRelevantInfo(
            ragQuery, g_RAGMaxRetrievedItems, g_RAGSimilarityThreshold, input.factualQuestion);
        FilterRagByLevel(results, ctx.botLevel);
        FilterRagByClass(results, ctx.botClass, ctx.playerClass);
        if (channel == SRC_GENERAL_LOCAL && !input.factualQuestion)
            FilterRagForCasualChatter(results);
        KnowledgeLevel level = OllamaPromptComposer::RollKnowledgeLevel(ctx, results);
        std::string bullets = g_RAGSystem->GetFormattedRAGInfo(results);
        input.knowledgeSection = OllamaPromptComposer::BuildKnowledgeSection(level, bullets);
        input.knowledgeLevel = level;
    }

    input.contextSection = BuildBotPromptContext(bot);
    input.nearbySection = GetBotCompactNearbySection(bot);

    PromptBundle bundle = OllamaPromptComposer::Build(PromptScenario::PlayerChat, ctx, input);
    EnrichPromptBundle(bundle, bot, ctx, player, channel);
    return bundle;
}

static void ApplyRagToInput(BotContext const& ctx, ScenarioInput& input, std::string const& query, bool factual,
    PromptScenario scenario)
{
    if (!g_EnableRAG || !g_RAGSystem)
        return;

    if (scenario == PromptScenario::RandomChatter && input.chatChannel == SRC_GENERAL_LOCAL)
        return;

    auto results = g_RAGSystem->RetrieveRelevantInfo(
        query, g_RAGMaxRetrievedItems, g_RAGSimilarityThreshold, factual);
    FilterRagByLevel(results, ctx.botLevel);
    FilterRagByClass(results, ctx.botClass, ctx.playerClass);
    if (input.chatChannel == SRC_GENERAL_LOCAL)
        FilterRagForCasualChatter(results);

    if (scenario == PromptScenario::EventReaction && input.chatChannel == SRC_GENERAL_LOCAL)
    {
        FilterRagToZoneOnly(results, ToLower(ctx.botZone));
        if (results.size() > 1)
            results.resize(1);
    }

    KnowledgeLevel level = OllamaPromptComposer::RollKnowledgeLevel(ctx, results);
    std::string bullets = g_RAGSystem->GetFormattedRAGInfo(results);
    input.knowledgeSection = OllamaPromptComposer::BuildKnowledgeSection(level, bullets);
    input.knowledgeLevel = level;
}

PromptBundle BuildRandomChatterPrompt(Player* bot, std::string const& environmentInfo, RandomIntent intent,
    ChatChannelSourceLocal channel)
{
    BotContext ctx = OllamaPromptComposer::GatherBotContext(bot, nullptr);
    ctx.personalityLine = GetPersonalityPromptForChannel(ctx.personalityKey, channel);
    ScenarioInput input;
    input.environmentSection = environmentInfo;
    input.randomIntent = intent;
    input.chatChannel = channel;

    std::string ragQuery = OllamaPromptComposer::BuildRagQuery(ctx.botZone + " " + ctx.botArea, ctx.botZone, ctx.botArea);
    ApplyRagToInput(ctx, input, ragQuery, false, PromptScenario::RandomChatter);
    input.contextSection = BuildBotPromptContext(bot);
    input.nearbySection = GetBotCompactNearbySection(bot);

    PromptBundle bundle = OllamaPromptComposer::Build(PromptScenario::RandomChatter, ctx, input);
    EnrichPromptBundle(bundle, bot, ctx, nullptr, channel, true);
    return bundle;
}

PromptBundle BuildEventReactionPrompt(Player* bot, Player* actorPlayer, std::string const& eventType,
    std::string const& eventDetail, std::string const& actorName, ChatChannelSourceLocal channel)
{
    BotContext ctx = OllamaPromptComposer::GatherBotContext(bot, actorPlayer);
    ctx.personalityLine = GetPersonalityPromptForChannel(ctx.personalityKey, channel);
    ScenarioInput input;
    input.eventType = eventType;
    input.eventDetail = eventDetail;
    input.actorName = actorName;
    input.chatChannel = channel;

    if (actorPlayer && g_EnableSentimentTracking)
        input.sentimentSection = GetSentimentPromptAddition(bot, actorPlayer);

    if (actorPlayer && g_EnableMemory)
    {
        std::string query = eventType + ": " + eventDetail;
        input.memorySection = GetMemoryPromptAddition(
            bot->GetGUID().GetRawValue(),
            actorPlayer->GetGUID().GetRawValue(),
            channel,
            query,
            actorPlayer->GetName());
    }

    std::string ragQuery = OllamaPromptComposer::BuildRagQuery(eventType + " " + eventDetail, ctx.botZone, ctx.botArea);
    ApplyRagToInput(ctx, input, ragQuery, false, PromptScenario::EventReaction);
    input.contextSection = BuildBotPromptContext(bot);

    PromptBundle bundle = OllamaPromptComposer::Build(PromptScenario::EventReaction, ctx, input);
    EnrichPromptBundle(bundle, bot, ctx, actorPlayer, channel);
    return bundle;
}
