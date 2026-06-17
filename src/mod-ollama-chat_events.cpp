#include "mod-ollama-chat_events.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_random.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat-utilities.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_memory.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_prompt.h"
#include "Player.h"
#include "ObjectAccessor.h"
#include "Guild.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "ChannelMgr.h"
#include "Channel.h"
#include "AiFactory.h"
#include "SpellMgr.h"
#include "AchievementMgr.h"
#include "GameObject.h"
#include <vector>
#include <thread>
#include <random>
#include <fmt/core.h>
#include <unordered_map>
#include <chrono>

static OllamaBotEventChatter eventChatter;
static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> botEventCooldowns;

namespace
{
PromptBundle BuildEventPromptBundle(Player* bot, Player* actorPlayer, std::string const& type,
    std::string const& detail, std::string const& actorName)
{
    if (!g_EventChatterPromptTemplate.empty())
    {
        PromptBundle bundle;
        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!ai)
            return bundle;

        std::string personality = GetBotPersonality(bot);
        std::string personalityPrompt = GetPersonalityPromptAddition(personality);
        AreaTableEntry const* botCurrentArea = ai->GetCurrentArea();
        AreaTableEntry const* botCurrentZone = ai->GetCurrentZone();
        std::string sentimentInfo;
        if (g_EnableSentimentTracking && actorPlayer)
            sentimentInfo = GetSentimentPromptAddition(bot, actorPlayer);

        bundle.user = SafeFormat(
            g_EventChatterPromptTemplate,
            fmt::arg("bot_name", bot->GetName()),
            fmt::arg("bot_level", bot->GetLevel()),
            fmt::arg("bot_class", ai->GetChatHelper() ? ai->GetChatHelper()->FormatClass(bot->getClass()) : "UnknownClass"),
            fmt::arg("bot_race", ai->GetChatHelper() ? ai->GetChatHelper()->FormatRace(bot->getRace()) : "UnknownRace"),
            fmt::arg("bot_gender", bot->getGender() == GENDER_MALE ? "Male" : "Female"),
            fmt::arg("bot_role", ai->GetChatHelper() ? ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot)) : "UnknownRole"),
            fmt::arg("bot_faction", bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde"),
            fmt::arg("bot_area", botCurrentArea ? ai->GetLocalizedAreaName(botCurrentArea) : "UnknownArea"),
            fmt::arg("bot_zone", botCurrentZone ? ai->GetLocalizedAreaName(botCurrentZone) : "UnknownZone"),
            fmt::arg("bot_map", bot->GetMap() ? bot->GetMap()->GetMapName() : "UnknownMap"),
            fmt::arg("bot_personality", personalityPrompt),
            fmt::arg("bot_personality_name", personality),
            fmt::arg("event_type", type),
            fmt::arg("event_detail", detail),
            fmt::arg("actor_name", actorName),
            fmt::arg("sentiment_info", sentimentInfo));
        if (g_EnableMemory && actorPlayer)
        {
            bundle.user += GetMemoryPromptAddition(
                bot->GetGUID().GetRawValue(),
                actorPlayer->GetGUID().GetRawValue(),
                type + ": " + detail,
                actorName) + "\n";
        }
        return bundle;
    }

    ChatChannelSourceLocal channel = bot->GetGroup() ? SRC_PARTY_LOCAL : SRC_GENERAL_LOCAL;
    return BuildEventReactionPrompt(bot, actorPlayer, type, detail, actorName, channel);
}
} // namespace

void OllamaBotEventChatter::DispatchGameEvent(Player* source, std::string type, std::string detail)
{
    if (!g_Enable || !g_EnableEventChatter)
        return;
    
    if (!source)
    {
       return;
    }

    bool isSourceBot = PlayerbotsMgr::instance().GetPlayerbotAI(source) != nullptr;
    bool hasNearbyRealPlayer = false;
    bool isGuildEvent = false;

    // Only set isGuildEvent for specific event types
    if (source->GetGuild() && g_EnableGuildEventChatter) {
        // List of event types that are considered guild events
        if (
            type == g_GuildEventTypeGuildJoin ||
            type == g_GuildEventTypeGuildLeave ||
            type == g_GuildEventTypeGuildPromotion ||
            type == g_GuildEventTypeGuildDemotion ||
            type == g_GuildEventTypeLevelUp ||
            type == g_GuildEventTypeEpicGear ||
            type == g_GuildEventTypeRareGear ||
            type == g_GuildEventTypeDungeonComplete ||
            type == g_GuildEventTypeGuildAchievement ||
            type == g_GuildEventTypeGuildLogin
        ) {
            // Check if there are real players in the guild
            Guild* guild = source->GetGuild();
            for (auto const& pair : ObjectAccessor::GetPlayers()) {
                Player* player = pair.second;
                if (!player || !player->IsInWorld())
                    continue;
                if (PlayerbotsMgr::instance().GetPlayerbotAI(player))
                    continue;
                if (player->GetGuild() && player->GetGuild()->GetId() == guild->GetId()) {
                    isGuildEvent = true;
                    break;
                }
            }
        }
    }

    // Check for nearby real players (original logic)
    for (auto const& pair : source->GetMap()->GetPlayers())
    {
        Player* player = pair.GetSource();
        if (player == source)
            continue;

        if (!PlayerbotsMgr::instance().GetPlayerbotAI(player) && player->IsWithinDist(source, g_EventChatterRealPlayerDistance, false))
        {
            hasNearbyRealPlayer = true;
            break;
        }
    }

    if (isSourceBot && !hasNearbyRealPlayer && !isGuildEvent)
    {
        if (g_DebugEnabled)
        {
            //LOG_INFO("server.loading", "[OllamaChat] Skipping bot source {} - no real players nearby", source->GetName());
        }
        return;
    }

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] DispatchGameEvent from {} | type={} | detail={}", source->GetName(), type, detail);

    float maxDist = g_EventChatterRealPlayerDistance;
    bool disableInCombat = g_DisableRepliesInCombat;

    std::vector<Player*> candidateBots;
    
    if (isGuildEvent)
    {
        // For guild events, get guild bots
        Guild* guild = source->GetGuild();
        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* player = pair.second;
            if (!player || !player->IsInWorld())
                continue;
                
            PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(player);
            if (!ai)
                continue;
                
            if (!player->GetGuild() || player->GetGuild()->GetId() != guild->GetId())
                continue;
                
            // For guild events, just check if there are any real players online in the guild
            bool hasRealPlayerInGuild = false;
            for (auto const& pair2 : ObjectAccessor::GetPlayers())
            {
                Player* guildPlayer = pair2.second;
                if (!guildPlayer || !guildPlayer->IsInWorld())
                    continue;
                    
                if (PlayerbotsMgr::instance().GetPlayerbotAI(guildPlayer))
                    continue;
                    
                if (guildPlayer->GetGuild() && guildPlayer->GetGuild()->GetId() == guild->GetId())
                {
                    hasRealPlayerInGuild = true;
                    break;
                }
            }
            
            if (hasRealPlayerInGuild)
                candidateBots.push_back(player);
        }
    }
    else
    {
        // For regular events, get nearby bots
        for (auto const& pair : source->GetMap()->GetPlayers())
        {
            Player* player = pair.GetSource();
            if (player->IsWithinDist(source, maxDist, false)) {
                candidateBots.push_back(player);
                if (g_DebugEnabled)
                    LOG_INFO("server.loading", "[OllamaChat] Nearby player {} within {:.1f} yards", player->GetName(), maxDist);
            }
        }
    }

    auto now = std::chrono::steady_clock::now();
    for (auto it = candidateBots.begin(); it != candidateBots.end(); )
    {
        uint64_t botGuid = (*it)->GetGUID().GetRawValue();
        auto cooldownIt = botEventCooldowns.find(botGuid);
        if (cooldownIt != botEventCooldowns.end() &&
            std::chrono::duration_cast<std::chrono::seconds>(now - cooldownIt->second).count() < g_EventCooldownTime)
        {
            it = candidateBots.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Check event-specific chance
    float eventChance = 0.0f;
    if (type == g_EventTypeLearnedSpell) {
        eventChance = g_EventTypeLearnedSpell_Chance;
    } else if (type == g_EventTypeDefeated) {
        eventChance = g_EventTypeDefeated_Chance;
    } else if (type == g_EventTypeDefeatedPlayer) {
        eventChance = g_EventTypeDefeatedPlayer_Chance;
    } else if (type == g_EventTypePetDefeated) {
        eventChance = g_EventTypePetDefeated_Chance;
    } else if (type == g_EventTypeGotItem) {
        eventChance = g_EventTypeGotItem_Chance;
    } else if (type == g_EventTypeDied) {
        eventChance = g_EventTypeDied_Chance;
    } else if (type == g_EventTypeCompletedQuest) {
        eventChance = g_EventTypeCompletedQuest_Chance;
    } else if (type == g_EventTypeRequestedDuel) {
        eventChance = g_EventTypeRequestedDuel_Chance;
    } else if (type == g_EventTypeStartedDueling) {
        eventChance = g_EventTypeStartedDueling_Chance;
    } else if (type == g_EventTypeWonDuel) {
        eventChance = g_EventTypeWonDuel_Chance;
    } else if (type == g_EventTypeLeveledUp) {
        eventChance = g_EventTypeLeveledUp_Chance;
    } else if (type == g_EventTypeAchievement) {
        eventChance = g_EventTypeAchievement_Chance;
    } else if (type == g_EventTypeUsedObject) {
        eventChance = g_EventTypeUsedObject_Chance;
    }

    // Add chance checks for all GuildEventType entries
    if (type == g_GuildEventTypeEpicGear) {
        eventChance = g_GuildEventTypeEpicGear_Chance;
    } else if (type == g_GuildEventTypeRareGear) {
        eventChance = g_GuildEventTypeRareGear_Chance;
    } else if (type == g_GuildEventTypeGuildJoin) {
        eventChance = g_GuildEventTypeGuildJoin_Chance;
    } else if (type == g_GuildEventTypeGuildLogin) {
        eventChance = g_GuildEventTypeGuildLogin_Chance;
    } else if (type == g_GuildEventTypeGuildLeave) {
        eventChance = g_GuildEventTypeGuildLeave_Chance;
    } else if (type == g_GuildEventTypeGuildPromotion) {
        eventChance = g_GuildEventTypeGuildPromotion_Chance;
    } else if (type == g_GuildEventTypeGuildDemotion) {
        eventChance = g_GuildEventTypeGuildDemotion_Chance;
    } else if (type == g_GuildEventTypeGuildAchievement) {
        eventChance = g_GuildEventTypeGuildAchievement_Chance;
    } else if (type == g_GuildEventTypeLevelUp) {
        eventChance = g_GuildEventTypeLevelUp_Chance;
    } else if (type == g_GuildEventTypeDungeonComplete) {
        eventChance = g_GuildEventTypeDungeonComplete_Chance;
    }

    if (urand(0, 100) > eventChance) {
        return;
    }

    uint32_t responses = 0;

    for (Player* bot : candidateBots)
    {
        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!ai) {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaChat] Skipping {} - not a bot", bot->GetName());
            continue;
        }

        if (disableInCombat && bot->IsInCombat()) {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaChat] Skipping {} - in combat", bot->GetName());
            continue;
        }

        uint32_t chance = 0;
        if(source == bot)
        {
            chance = g_EventChatterBotSelfCommentChance;
        } 
        else if (isGuildEvent)
        {
            chance = g_GuildChatterBotCommentChance;
        }
        else 
        {
            chance = g_EventChatterBotCommentChance;
        }

        if (urand(1, 100) > chance) {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaChat] Skipping {} - failed chance roll", bot->GetName());
            continue;
        }

        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[OllamaChat] Queueing event for bot {}", bot->GetName());

        botEventCooldowns[bot->GetGUID().GetRawValue()] = now;
        QueueEvent(bot, type, detail, source->GetName(), isGuildEvent);

        responses++;
        uint32_t maxBots = isGuildEvent ? g_GuildChatterMaxBotsPerEvent : g_EventChatterMaxBotsPerPlayer;
        if (maxBots > 0 && responses >= maxBots)
            break;
    }

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] Dispatch complete. {} bots responded", responses);
}


void OllamaBotEventChatter::QueueEvent(Player* bot, std::string type, std::string detail, std::string actorName, bool isGuildEvent)
{
    if (!g_Enable || !g_EnableEventChatter || !bot)
        return;

    uint64_t botGuid = bot->GetGUID().GetRawValue();

    std::thread([this, botGuid, type, detail, actorName, isGuildEvent]()
    {
        try
        {
            Player* botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
            if (!botPtr) return;

            Player* actorPlayer = actorName.empty() ? nullptr : ObjectAccessor::FindPlayerByName(actorName);
            PromptBundle bundle = BuildEventPromptBundle(botPtr, actorPlayer, type, detail, actorName);
            if (bundle.user.empty()) return;

            auto responseFuture = SubmitQuery(bundle);
            if (!responseFuture.valid())
                return;
            std::string response = responseFuture.get();
            if (response.empty())
            {
                if (g_DebugEnabled)
                    LOG_INFO("server.loading", "[OllamaChat] Bot {} skipped event response due to API error", botPtr->GetName());
                return;
            }

            // reacquire pointers before use
            botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
            if (!botPtr) return;
            PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
            if (!botAI) return;

            // Simulate typing delay if enabled
            if (g_EnableTypingSimulation)
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
            }

            // Route response to random appropriate channel
            if (isGuildEvent && botPtr->GetGuild())
            {
                // Check if guild chat is disabled
                if (g_DisableForGuild)
                {
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Guild event chatter skipped (guild channels disabled)");
                    return;
                }
                
                botAI->SayToGuild(response);
                ProcessBotChatMessage(botPtr, response, SRC_GUILD_LOCAL, nullptr);
            }
            else if (botPtr->GetGroup())
            {
                // Check if party chat is disabled
                if (g_DisableForParty)
                {
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Party event chatter skipped (party channels disabled)");
                    return;
                }
                
                botAI->SayToParty(response);
                ProcessBotChatMessage(botPtr, response, SRC_PARTY_LOCAL, nullptr);
            }
            else
            {
                // For solo bots, randomly pick between Say and General channel
                std::vector<std::string> channels;
                
                // Only add General if custom channels are not disabled
                if (!g_DisableForCustomChannels)
                {
                    channels.push_back("General");
                }
                
                // Only add Say if not disabled
                if (!g_DisableForSayYell)
                {
                    channels.push_back("Say");
                }
                
                // If no channels are available, skip event chatter
                if (channels.empty())
                {
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping event chatter (all available channels disabled)", botPtr->GetName());
                    return;
                }
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> dist(0, channels.size() - 1);
                std::string selectedChannel = channels[dist(gen)];
                
                if (selectedChannel == "Say")
                {
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Bot Event Chatter Say: {}", response);
                    botAI->Say(response);
                    ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                }
                else if (selectedChannel == "General")
                {
                    EnsureBotInGeneralChannel(botPtr);
                    Channel* generalChannel = nullptr;
                    if (ChannelMgr* cMgr = ChannelMgr::forTeam(botPtr->GetTeamId()))
                        generalChannel = cMgr->GetChannel("General", botPtr);

                    if (generalChannel)
                    {
                        TrySendGeneralChat(botPtr, response, "", generalChannel);
                        return;
                    }

                    botAI->Say(response);
                    ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                }
            }

            if (g_EnableMemory && !actorName.empty() && botPtr)
            {
                Player* actorPlayer = ObjectAccessor::FindPlayerByName(actorName);
                if (actorPlayer && !PlayerbotsMgr::instance().GetPlayerbotAI(actorPlayer))
                {
                    std::string ctx = type + ": " + detail;
                    AppendBotConversation(botPtr->GetGUID().GetRawValue(),
                        actorPlayer->GetGUID().GetRawValue(), ctx, response, true);
                }
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("server.loading", "[OllamaChat] Exception in QueueEvent thread: {}", e.what());
        }
    }).detach();
}


// === Script Hooks ===

ChatOnKill::ChatOnKill() : PlayerScript("ChatOnKill") {}

void ChatOnKill::OnPlayerCreatureKill(Player* killer, Creature* victim)
{
    if (!killer || !victim)
    {
        return;
    }
    eventChatter.DispatchGameEvent(killer, g_EventTypeDefeated, victim->GetName());
}

void ChatOnKill::OnPlayerPVPKill(Player* killer, Player* killed)
{
    if (!killer || !killed)
    {
        return;
    }
    eventChatter.DispatchGameEvent(killer, g_EventTypeDefeatedPlayer, killed->GetName());
}

void ChatOnKill::OnPlayerCreatureKilledByPet(Player* owner, Creature* victim)
{
    if (!owner || !victim)
    {
        return;
    }
    eventChatter.DispatchGameEvent(owner, g_EventTypePetDefeated, victim->GetName());
}

ChatOnLoot::ChatOnLoot() : PlayerScript("ChatOnLoot") {}

void ChatOnLoot::OnPlayerStoreNewItem(Player* player, Item* item, uint32 /*count*/)
{
    if (!player || !item || !item->GetTemplate())
    {
        return;
    }
    if (item->GetTemplate()->Quality >= ITEM_QUALITY_UNCOMMON)
    {
        eventChatter.DispatchGameEvent(player, g_EventTypeGotItem, item->GetTemplate()->Name1);
    }
    
    // Guild-specific gear events
    if (player->GetGuild() && g_EnableGuildEventChatter)
    {
        if (item->GetTemplate()->Quality == ITEM_QUALITY_EPIC && !g_GuildEventTypeEpicGear.empty())
        {
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeEpicGear, item->GetTemplate()->Name1);
        }
        else if (item->GetTemplate()->Quality == ITEM_QUALITY_RARE && !g_GuildEventTypeRareGear.empty())
        {
            // Only announce rare gear if it's equipment (not consumables, etc.)
            if (item->GetTemplate()->Class == ITEM_CLASS_WEAPON || 
                item->GetTemplate()->Class == ITEM_CLASS_ARMOR)
            {
                eventChatter.DispatchGameEvent(player, g_GuildEventTypeRareGear, item->GetTemplate()->Name1);
            }
        }
    }
}

ChatOnDeath::ChatOnDeath() : PlayerScript("ChatOnDeath") {}

void ChatOnDeath::OnPlayerJustDied(Player* player)
{
    if (!player)
    {
        return;
    }
    eventChatter.DispatchGameEvent(player, g_EventTypeDied, "");
}

ChatOnQuest::ChatOnQuest() : PlayerScript("ChatOnQuest") {}

void ChatOnQuest::OnPlayerCompleteQuest(Player* player, Quest const* quest)
{
    if (!player || !quest)
    {
        return;
    }
    eventChatter.DispatchGameEvent(player, g_EventTypeCompletedQuest, quest->GetTitle());
    
    // Guild-specific dungeon completion events
    if (player->GetGuild() && g_EnableGuildEventChatter && !g_GuildEventTypeDungeonComplete.empty())
    {
        // Check if this is a dungeon quest
        if (player->GetMap() && player->GetMap()->IsDungeon())
        {
            std::string dungeonInfo = SafeFormat("{} in {}", quest->GetTitle(), player->GetMap()->GetMapName());
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeDungeonComplete, dungeonInfo);
        }
    }
}

ChatOnLearn::ChatOnLearn() : PlayerScript("ChatOnLearn") {}

void ChatOnLearn::OnPlayerLearnSpell(Player* player, uint32 spellID)
{
    if (!player)
    {
        return;
    }
    SpellInfo const* spellInfo = sSpellMgr->GetSpellInfo(spellID);
    if (spellInfo)
    {
        eventChatter.DispatchGameEvent(player, g_EventTypeLearnedSpell, spellInfo->SpellName[0]);
    }
    else
    {
        eventChatter.DispatchGameEvent(player, g_EventTypeLearnedSpell, std::to_string(spellID));
    }
}

ChatOnDuel::ChatOnDuel() : PlayerScript("ChatOnDuel") {}

void ChatOnDuel::OnPlayerDuelRequest(Player* target, Player* challenger)
{
    if (!challenger || !target)
    {
        return;
    }
    eventChatter.DispatchGameEvent(challenger, g_EventTypeRequestedDuel, target->GetName());
}

void ChatOnDuel::OnPlayerDuelStart(Player* player1, Player* player2)
{
    if (!player1 || !player2)
    {
        return;
    }
    eventChatter.DispatchGameEvent(player1, g_EventTypeStartedDueling, player2->GetName());
}

void ChatOnDuel::OnPlayerDuelEnd(Player* winner, Player* loser, DuelCompleteType /*type*/)
{
    if (!winner || !loser)
    {
        return;
    }
    eventChatter.DispatchGameEvent(winner, g_EventTypeWonDuel, loser->GetName());
}

ChatOnLevelUp::ChatOnLevelUp() : PlayerScript("ChatOnLevelUp") {}

void ChatOnLevelUp::OnPlayerLevelChanged(Player* player, uint8 /*oldLevel*/)
{
    if (!player)
    {
        return;
    }
    eventChatter.DispatchGameEvent(player, g_EventTypeLeveledUp, std::to_string(player->GetLevel()));
    
    // Guild-specific level up events
    if (player->GetGuild() && g_EnableGuildEventChatter && !g_GuildEventTypeLevelUp.empty())
    {
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeLevelUp, std::to_string(player->GetLevel()));
    }
}

ChatOnAchievement::ChatOnAchievement() : PlayerScript("ChatOnAchievement") {}


void ChatOnAchievement::OnPlayerAchievementComplete(Player* player, AchievementEntry const* achievement)
{
    if (!player || !achievement)
    {
        return;
    }
    eventChatter.DispatchGameEvent(player, g_EventTypeAchievement, achievement->name[0]);

    // Guild-specific achievement event for real players only
    if (player->GetGuild() && g_EnableGuildEventChatter && !g_GuildEventTypeGuildAchievement.empty())
    {
        if (!PlayerbotsMgr::instance().GetPlayerbotAI(player)) // Only real players
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildAchievement, achievement->name[0]);
    }
}

ChatOnGameObjectUse::ChatOnGameObjectUse() : AllGameObjectScript("ChatOnGameObjectUse") {}

bool ChatOnGameObjectUse::CanGameObjectGossipHello(Player* player, GameObject* go)
{
    if (!player || !go || !go->GetGOInfo())
        return false;

    eventChatter.DispatchGameEvent(player, g_EventTypeUsedObject, go->GetGOInfo()->name);
    return false;
}

ChatOnGuildMemberChange::ChatOnGuildMemberChange() : GuildScript("ChatOnGuildMemberChange", {
    GUILDHOOK_ON_ADD_MEMBER,
    GUILDHOOK_ON_REMOVE_MEMBER,
    GUILDHOOK_ON_EVENT,
}) {}

void ChatOnGuildMemberChange::OnAddMember(Guild* guild, Player* player, uint8& /*plRank*/)
{
    if (!player || !guild || !g_EnableGuildEventChatter)
        return;

    if (!g_GuildEventTypeGuildJoin.empty())
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildJoin, guild->GetName());
}

void ChatOnGuildMemberChange::OnRemoveMember(Guild* guild, Player* player, bool /*isDisbanding*/, bool /*isKicked*/)
{
    if (!player || !guild || !g_EnableGuildEventChatter)
        return;

    if (!g_GuildEventTypeGuildLeave.empty())
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildLeave, guild->GetName());
}

void ChatOnGuildMemberChange::OnEvent(Guild* guild, uint8 eventType, ObjectGuid::LowType playerGuid1, ObjectGuid::LowType /*playerGuid2*/, uint8 newRank)
{
    if (!guild || !g_EnableGuildEventChatter)
        return;

    Player* player = ObjectAccessor::FindPlayerByLowGUID(playerGuid1);
    if (!player)
        return;

    if (eventType == GUILD_EVENT_LOG_PROMOTE_PLAYER)
    {
        if (!g_GuildEventTypeGuildPromotion.empty())
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildPromotion, std::to_string(newRank));
    }
    else if (eventType == GUILD_EVENT_LOG_DEMOTE_PLAYER)
    {
        if (!g_GuildEventTypeGuildDemotion.empty())
            eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildDemotion, std::to_string(newRank));
    }
}

ChatOnGuildLogin::ChatOnGuildLogin() : PlayerScript("ChatOnGuildLogin") {}

void ChatOnGuildLogin::OnPlayerLogin(Player* player)
{
    if (!player || !player->GetGuild() || !g_EnableGuildEventChatter)
        return;
    if (PlayerbotsMgr::instance().GetPlayerbotAI(player))
        return;
    if (!g_GuildEventTypeGuildLogin.empty())
        eventChatter.DispatchGameEvent(player, g_GuildEventTypeGuildLogin, player->GetGuild()->GetName());
}

