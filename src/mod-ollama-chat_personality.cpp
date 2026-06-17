#include "mod-ollama-chat_personality.h"
#include "Player.h"
#include "PlayerbotMgr.h"
#include "Log.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_handler.h"
#include "DatabaseEnv.h"
#include <random>
#include <vector>

namespace
{
char const* kGeneralCasualPrompt =
    "Casual Wrath zone chat: first person, abbreviations OK, never wiki-tone or narration.";

bool ShouldClampGeneralPersonality(std::string const& key)
{
    static char const* kClamp[] = {
        "NPC_IMPERSONATOR", "MENTOR", "SCHOLAR", "BARD", "ANCIENT_WISE_ONE",
        "HEROIC_LEADER", "POET", "GLITCHED_AI", "ROLEPLAYER"
    };
    for (char const* k : kClamp)
    {
        if (key == k)
            return true;
    }
    return false;
}

std::string GetPersonalityToneHint(std::string const& key)
{
    static char const* kHints[][2] = {
        { "TRICKSTER", "wry and sarcastic" },
        { "GAMER", "min-max and efficiency focused" },
        { "GOBLIN_MERCHANT", "greedy and business-minded" },
        { "CASUAL", "relaxed and friendly" },
        { "GRUMPY_VETERAN", "grumpy but helpful" },
        { "LOOTGOBLIN", "loot and gold obsessed" },
        { "PVP_HARDCORE", "pvp focused" },
        { "RAIDER", "raid focused" },
        { "TRADER", "economy focused" },
        { "LONE_WOLF", "short and direct" },
        { "FOOL", "clueless but eager" },
        { "CONSPIRACY_THEORIST", "suspicious of rumors" },
        { "EDGE_LORD", "dark and brooding" },
        { "FANATIC", "faction obsessed" },
        { "HYPE_MAN", "overhyped" },
        { "PARANOID", "paranoid" },
        { "FLIRT", "flirty" },
        { "RAGER", "angry" },
        { "STONER", "chill" },
        { "YOUNG_APPRENTICE", "eager newbie" },
        { "WANNABE_VILLAIN", "villain vibes" },
        { "JOLLY_BEER_LOVER", "drunk dwarf vibes" },
        { "PIRATE", "pirate slang" },
        { "CHEF", "food obsessed" },
    };
    for (auto const& row : kHints)
    {
        if (key == row[0])
            return std::string("Tone hint: ") + row[1] + ". Answer the ask first; flavor second.";
    }
    return "Tone hint: stay in character lightly. Answer the ask first; flavor second.";
}
} // namespace

std::string GetPersonalityPromptForChannel(const std::string& type, ChatChannelSourceLocal channel)
{
    if (channel == SRC_GENERAL_LOCAL)
    {
        if (ShouldClampGeneralPersonality(type))
            return kGeneralCasualPrompt;
        return GetPersonalityToneHint(type);
    }
    return GetPersonalityPromptAddition(type);
}

// Internal personality map
std::string GetBotPersonality(Player* bot)
{
    uint64_t botGuid = bot->GetGUID().GetRawValue();

    // If personality already assigned, return it (but only if RP personalities are enabled)
    auto it = g_BotPersonalityList.find(botGuid);
    if (it != g_BotPersonalityList.end())
    {
        // If RP personalities are disabled, reset to default
        if (!g_EnableRPPersonalities)
        {
            g_BotPersonalityList[botGuid] = "default";
            return "default";
        }
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Using existing personality '{}' for bot {}", it->second, bot->GetName());
        }
        return it->second;
    }

    // RP personalities disabled or config not loaded
    if (!g_EnableRPPersonalities || g_PersonalityKeysRandomOnly.empty())
    {
        g_BotPersonalityList[botGuid] = "default";
        return "default";
    }

    // Try to load from database if you have persistence
    if (g_BotPersonalityList.find(botGuid) != g_BotPersonalityList.end())
    {
        // DB stores string keys now
        std::string dbPersonality = g_BotPersonalityList[botGuid];

        if (dbPersonality.empty())
        {
            dbPersonality = "default";
        }

        g_BotPersonalityList[botGuid] = dbPersonality;

        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Using database personality '{}' for bot {}", dbPersonality, bot->GetName());
        }
        return dbPersonality;
    }

    // Otherwise, assign randomly from config (only from non-manual personalities)
    if (g_PersonalityKeysRandomOnly.empty())
        return "default";

    uint32 newIdx = urand(0, g_PersonalityKeysRandomOnly.size() - 1);
    std::string chosenPersonality = g_PersonalityKeysRandomOnly[newIdx];
    g_BotPersonalityList[botGuid] = chosenPersonality;

    // Save to database if schema supports string (recommend TEXT or VARCHAR column for personality)
    QueryResult tableExists = CharacterDatabase.Query(
        "SELECT * FROM information_schema.tables WHERE table_schema = 'acore_characters' AND table_name = 'mod_ollama_chat_personality' LIMIT 1;");
    if (!tableExists)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Please source the required database table first");
    }
    else
    {
        CharacterDatabase.Execute("INSERT INTO mod_ollama_chat_personality (guid, personality) VALUES ({}, '{}')", botGuid, chosenPersonality);
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Assigned new personality '{}' to bot {}", chosenPersonality, bot->GetName());
    }
    return chosenPersonality;
}


std::string GetPersonalityPromptAddition(const std::string& personality)
{
    auto it = g_PersonalityPrompts.find(personality);
    if (it != g_PersonalityPrompts.end())
        return it->second;
    return g_DefaultPersonalityPrompt;
}

bool SetBotPersonality(Player* bot, const std::string& personality)
{
    if (!bot)
        return false;
    
    uint64_t botGuid = bot->GetGUID().GetRawValue();
    
    // Check if personality exists
    if (g_PersonalityPrompts.find(personality) == g_PersonalityPrompts.end() && personality != "default")
    {
        return false;
    }
    
    // Update in memory
    g_BotPersonalityList[botGuid] = personality;
    
    // Update in database
    CharacterDatabase.Execute("REPLACE INTO mod_ollama_chat_personality (guid, personality) VALUES ({}, '{}')", 
                             botGuid, personality);
    
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Set personality '{}' for bot {}", personality, bot->GetName());
    }
    
    return true;
}

std::vector<std::string> GetAllPersonalityKeys()
{
    return g_PersonalityKeys;
}

bool PersonalityExists(const std::string& personality)
{
    if (personality == "default")
        return true;
    return g_PersonalityPrompts.find(personality) != g_PersonalityPrompts.end();
}

void ClearAllBotPersonalities()
{
    g_BotPersonalityList.clear();
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Cleared all bot personality assignments due to RP personalities being disabled");
    }
}
