#include "mod-ollama-chat_events.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_random.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat-utilities.h"
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
#include "SpellMgr.h"
#include "AchievementMgr.h"
#include "GameObject.h"
#include <vector>
#include <thread>
#include <random>
#include <unordered_map>
#include <chrono>

static OllamaBotEventChatter eventChatter;
static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> botEventCooldowns;

char const* EventKindLabel(EventKind kind)
{
    switch (kind)
    {
        case EventKind::Defeated:            return "defeated";
        case EventKind::DefeatedPlayer:      return "defeated_player";
        case EventKind::PetDefeated:         return "pet_defeated";
        case EventKind::GotItem:             return "got_item";
        case EventKind::Died:                return "died";
        case EventKind::CompletedQuest:      return "completed_quest";
        case EventKind::LearnedSpell:        return "learned_spell";
        case EventKind::RequestedDuel:       return "requested_duel";
        case EventKind::StartedDueling:      return "started_dueling";
        case EventKind::WonDuel:             return "won_duel";
        case EventKind::LeveledUp:           return "leveled_up";
        case EventKind::Achievement:         return "achievement";
        case EventKind::UsedObject:          return "used_object";
        case EventKind::GuildEpicGear:       return "guild_epic_gear";
        case EventKind::GuildRareGear:       return "guild_rare_gear";
        case EventKind::GuildJoin:           return "guild_join";
        case EventKind::GuildLeave:          return "guild_leave";
        case EventKind::GuildPromotion:      return "guild_promotion";
        case EventKind::GuildDemotion:       return "guild_demotion";
        case EventKind::GuildLogin:          return "guild_login";
        case EventKind::GuildAchievement:    return "guild_achievement";
        case EventKind::GuildLevelUp:        return "guild_level_up";
        case EventKind::GuildDungeonComplete: return "guild_dungeon_complete";
        default:                             return "event";
    }
}

std::string FormatEventHistoryContext(EventKind kind, std::string const& display)
{
    // Stable sentinel so load can mark isEvent without conf-string heuristics.
    return std::string("evt|") + EventKindLabel(kind) + ": " + display;
}

bool IsGuildEventKind(EventKind kind)
{
    return kind >= EventKind::GuildEpicGear && kind < EventKind::COUNT;
}

namespace
{
bool SameGroup(Player* a, Player* b)
{
    if (!a || !b)
        return false;
    if (a == b)
        return true;
    return a->GetGroup() && b->GetGroup() && a->GetGroup() == b->GetGroup();
}

bool RequiresParty(EventKind kind)
{
    switch (kind)
    {
        case EventKind::GotItem:
        case EventKind::CompletedQuest:
        case EventKind::UsedObject:
        case EventKind::LearnedSpell:
        case EventKind::Achievement:
        case EventKind::LeveledUp:
            return true;
        default:
            return false;
    }
}

float EventChance(EventKind kind)
{
    switch (kind)
    {
        case EventKind::LearnedSpell:         return static_cast<float>(g_EventTypeLearnedSpell_Chance);
        case EventKind::Defeated:             return static_cast<float>(g_EventTypeDefeated_Chance);
        case EventKind::DefeatedPlayer:       return static_cast<float>(g_EventTypeDefeatedPlayer_Chance);
        case EventKind::PetDefeated:          return static_cast<float>(g_EventTypePetDefeated_Chance);
        case EventKind::GotItem:              return static_cast<float>(g_EventTypeGotItem_Chance);
        case EventKind::Died:                 return static_cast<float>(g_EventTypeDied_Chance);
        case EventKind::CompletedQuest:       return static_cast<float>(g_EventTypeCompletedQuest_Chance);
        case EventKind::RequestedDuel:        return static_cast<float>(g_EventTypeRequestedDuel_Chance);
        case EventKind::StartedDueling:       return static_cast<float>(g_EventTypeStartedDueling_Chance);
        case EventKind::WonDuel:              return static_cast<float>(g_EventTypeWonDuel_Chance);
        case EventKind::LeveledUp:            return static_cast<float>(g_EventTypeLeveledUp_Chance);
        case EventKind::Achievement:          return static_cast<float>(g_EventTypeAchievement_Chance);
        case EventKind::UsedObject:           return static_cast<float>(g_EventTypeUsedObject_Chance);
        case EventKind::GuildEpicGear:        return static_cast<float>(g_GuildEventTypeEpicGear_Chance);
        case EventKind::GuildRareGear:        return static_cast<float>(g_GuildEventTypeRareGear_Chance);
        case EventKind::GuildJoin:            return static_cast<float>(g_GuildEventTypeGuildJoin_Chance);
        case EventKind::GuildLogin:           return static_cast<float>(g_GuildEventTypeGuildLogin_Chance);
        case EventKind::GuildLeave:           return static_cast<float>(g_GuildEventTypeGuildLeave_Chance);
        case EventKind::GuildPromotion:       return static_cast<float>(g_GuildEventTypeGuildPromotion_Chance);
        case EventKind::GuildDemotion:        return static_cast<float>(g_GuildEventTypeGuildDemotion_Chance);
        case EventKind::GuildAchievement:     return static_cast<float>(g_GuildEventTypeGuildAchievement_Chance);
        case EventKind::GuildLevelUp:         return static_cast<float>(g_GuildEventTypeLevelUp_Chance);
        case EventKind::GuildDungeonComplete: return static_cast<float>(g_GuildEventTypeDungeonComplete_Chance);
        default:                              return 0.0f;
    }
}

bool GuildHasRealPlayer(Guild* guild)
{
    if (!guild)
        return false;
    uint32 const guildId = guild->GetId();
    for (auto const& pair : ObjectAccessor::GetPlayers())
    {
        Player* player = pair.second;
        if (!player || !player->IsInWorld())
            continue;
        if (PlayerbotsMgr::instance().GetPlayerbotAI(player))
            continue;
        if (player->GetGuild() && player->GetGuild()->GetId() == guildId)
            return true;
    }
    return false;
}
} // namespace

void OllamaBotEventChatter::DispatchGameEvent(Player* source, EventKind kind, EventDetail const& detail)
{
    if (!g_Enable || !g_EnableEventChatter || !source)
        return;

    bool const isSourceBot = PlayerbotsMgr::instance().GetPlayerbotAI(source) != nullptr;
    bool hasNearbyRealPlayer = false;
    bool isGuildEvent = false;

    if (IsGuildEventKind(kind) && source->GetGuild() && g_EnableGuildEventChatter)
        isGuildEvent = GuildHasRealPlayer(source->GetGuild());

    if (source->GetMap())
    {
        for (auto const& pair : source->GetMap()->GetPlayers())
        {
            Player* player = pair.GetSource();
            if (player == source)
                continue;
            if (!PlayerbotsMgr::instance().GetPlayerbotAI(player) &&
                player->IsWithinDist(source, g_EventChatterRealPlayerDistance, false))
            {
                hasNearbyRealPlayer = true;
                break;
            }
        }
    }

    if (isSourceBot && !hasNearbyRealPlayer && !isGuildEvent)
        return;

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] DispatchGameEvent from {} | kind={} | detail={}",
            source->GetName(), EventKindLabel(kind), detail.display);

    float const maxDist = g_EventChatterRealPlayerDistance;
    bool const disableInCombat = g_DisableRepliesInCombat;
    std::vector<Player*> candidateBots;

    if (isGuildEvent)
    {
        Guild* guild = source->GetGuild();
        uint32 const guildId = guild->GetId();
        bool const hasReal = GuildHasRealPlayer(guild);
        if (!hasReal)
            return;
        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* player = pair.second;
            if (!player || !player->IsInWorld())
                continue;
            if (!PlayerbotsMgr::instance().GetPlayerbotAI(player))
                continue;
            if (!player->GetGuild() || player->GetGuild()->GetId() != guildId)
                continue;
            candidateBots.push_back(player);
        }
    }
    else if (source->GetMap())
    {
        for (auto const& pair : source->GetMap()->GetPlayers())
        {
            Player* player = pair.GetSource();
            if (player->IsWithinDist(source, maxDist, false))
                candidateBots.push_back(player);
        }
    }

    auto now = std::chrono::steady_clock::now();
    for (auto it = candidateBots.begin(); it != candidateBots.end(); )
    {
        uint64_t botGuid = (*it)->GetGUID().GetRawValue();
        auto cooldownIt = botEventCooldowns.find(botGuid);
        if (cooldownIt != botEventCooldowns.end() &&
            std::chrono::duration_cast<std::chrono::seconds>(now - cooldownIt->second).count() < g_EventCooldownTime)
            it = candidateBots.erase(it);
        else
            ++it;
    }

    float const eventChance = EventChance(kind);
    if (eventChance <= 0.0f)
        return;
    if (urand(1, 100) > static_cast<uint32_t>(eventChance))
        return;

    uint32_t responses = 0;
    for (Player* bot : candidateBots)
    {
        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!ai)
            continue;
        if (disableInCombat && bot->IsInCombat())
            continue;
        if (!isGuildEvent && RequiresParty(kind) && !SameGroup(source, bot))
            continue;

        uint32_t chance = 0;
        if (source == bot)
            chance = g_EventChatterBotSelfCommentChance;
        else if (isGuildEvent)
            chance = g_GuildChatterBotCommentChance;
        else
            chance = g_EventChatterBotCommentChance;

        if (urand(1, 100) > chance)
            continue;

        botEventCooldowns[bot->GetGUID().GetRawValue()] = now;
        QueueEvent(bot, kind, detail, source->GetName(), isGuildEvent);

        ++responses;
        uint32_t maxBots = isGuildEvent ? g_GuildChatterMaxBotsPerEvent : g_EventChatterMaxBotsPerPlayer;
        if (maxBots > 0 && responses >= maxBots)
            break;
    }
}

void OllamaBotEventChatter::QueueEvent(Player* bot, EventKind kind, EventDetail const& detail,
    std::string actorName, bool isGuildEvent)
{
    if (!g_Enable || !g_EnableEventChatter || !bot)
        return;

    uint64_t botGuid = bot->GetGUID().GetRawValue();
    EventDetail detailCopy = detail;

    std::thread([botGuid, kind, detailCopy, actorName, isGuildEvent]()
    {
        try
        {
            Player* botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
            if (!botPtr)
                return;

            Player* actorPlayer = actorName.empty() ? nullptr : ObjectAccessor::FindPlayerByName(actorName);
            ChatChannelSourceLocal channel = botPtr->GetGroup() ? SRC_PARTY_LOCAL : SRC_GENERAL_LOCAL;
            PromptBundle bundle = BuildEventReactionPrompt(botPtr, actorPlayer,
                EventKindLabel(kind), detailCopy.display, EventTaskFor(kind), actorName, channel);
            if (bundle.user.empty())
                return;

            auto responseFuture = SubmitQuery(bundle);
            if (!responseFuture.valid())
                return;
            std::string response = responseFuture.get();
            if (response.empty())
                return;

            botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
            if (!botPtr)
                return;
            PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
            if (!botAI)
                return;

            if (g_EnableTypingSimulation)
            {
                uint32_t delay = g_TypingSimulationBaseDelay + (response.length() * g_TypingSimulationDelayPerChar);
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                if (!botPtr)
                    return;
                botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                if (!botAI)
                    return;
            }

            if (isGuildEvent && botPtr->GetGuild())
            {
                if (g_DisableForGuild)
                    return;
                botAI->SayToGuild(response);
                ProcessBotChatMessage(botPtr, response, SRC_GUILD_LOCAL, nullptr);
            }
            else if (botPtr->GetGroup())
            {
                if (g_DisableForParty)
                    return;
                botAI->SayToParty(response);
                ProcessBotChatMessage(botPtr, response, SRC_PARTY_LOCAL, nullptr);
            }
            else
            {
                std::vector<std::string> channels;
                if (!g_DisableForCustomChannels)
                    channels.push_back("General");
                if (!g_DisableForSayYell)
                    channels.push_back("Say");
                if (channels.empty())
                    return;

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> dist(0, channels.size() - 1);
                std::string selectedChannel = channels[dist(gen)];

                if (selectedChannel == "Say")
                {
                    botAI->Say(response);
                    ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                }
                else
                {
                    EnsureBotInGeneralChannel(botPtr);
                    Channel* generalChannel = nullptr;
                    if (ChannelMgr* cMgr = ChannelMgr::forTeam(botPtr->GetTeamId()))
                        generalChannel = cMgr->GetChannel("General", botPtr);
                    if (generalChannel)
                    {
                        TrySendGeneralChat(botPtr, response, "", generalChannel);
                    }
                    else
                    {
                        botAI->Say(response);
                        ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                    }
                }
            }

            if (g_EnableMemory && !actorName.empty() && botPtr)
            {
                Player* actor = ObjectAccessor::FindPlayerByName(actorName);
                if (actor && !PlayerbotsMgr::instance().GetPlayerbotAI(actor))
                {
                    std::string ctx = FormatEventHistoryContext(kind, detailCopy.display);
                    AppendBotConversation(botPtr->GetGUID().GetRawValue(),
                        actor->GetGUID().GetRawValue(), ctx, response, true);
                }
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("server.loading", "[OllamaChat] Exception in QueueEvent thread: {}", e.what());
        }
    }).detach();
}

ChatOnKill::ChatOnKill() : PlayerScript("ChatOnKill") {}

void ChatOnKill::OnPlayerCreatureKill(Player* killer, Creature* victim)
{
    if (!killer || !victim)
        return;
    eventChatter.DispatchGameEvent(killer, EventKind::Defeated, { victim->GetName(), victim->GetEntry() });
}

void ChatOnKill::OnPlayerPVPKill(Player* killer, Player* killed)
{
    if (!killer || !killed)
        return;
    eventChatter.DispatchGameEvent(killer, EventKind::DefeatedPlayer, { killed->GetName(), 0 });
}

void ChatOnKill::OnPlayerCreatureKilledByPet(Player* owner, Creature* victim)
{
    if (!owner || !victim)
        return;
    eventChatter.DispatchGameEvent(owner, EventKind::PetDefeated, { victim->GetName(), victim->GetEntry() });
}

ChatOnLoot::ChatOnLoot() : PlayerScript("ChatOnLoot") {}

void ChatOnLoot::OnPlayerStoreNewItem(Player* player, Item* item, uint32 /*count*/)
{
    if (!player || !item || !item->GetTemplate())
        return;

    ItemTemplate const* tmpl = item->GetTemplate();
    uint32 const entry = item->GetEntry();

    if (tmpl->Quality >= ITEM_QUALITY_UNCOMMON)
        eventChatter.DispatchGameEvent(player, EventKind::GotItem, { tmpl->Name1, entry });

    if (player->GetGuild() && g_EnableGuildEventChatter)
    {
        if (tmpl->Quality == ITEM_QUALITY_EPIC)
            eventChatter.DispatchGameEvent(player, EventKind::GuildEpicGear, { tmpl->Name1, entry });
        else if (tmpl->Quality == ITEM_QUALITY_RARE &&
            (tmpl->Class == ITEM_CLASS_WEAPON || tmpl->Class == ITEM_CLASS_ARMOR))
            eventChatter.DispatchGameEvent(player, EventKind::GuildRareGear, { tmpl->Name1, entry });
    }
}

ChatOnDeath::ChatOnDeath() : PlayerScript("ChatOnDeath") {}

void ChatOnDeath::OnPlayerJustDied(Player* player)
{
    if (!player)
        return;
    eventChatter.DispatchGameEvent(player, EventKind::Died, {});
}

ChatOnQuest::ChatOnQuest() : PlayerScript("ChatOnQuest") {}

void ChatOnQuest::OnPlayerCompleteQuest(Player* player, Quest const* quest)
{
    if (!player || !quest)
        return;
    eventChatter.DispatchGameEvent(player, EventKind::CompletedQuest, { quest->GetTitle(), quest->GetQuestId() });

    if (player->GetGuild() && g_EnableGuildEventChatter && player->GetMap() && player->GetMap()->IsDungeon())
    {
        std::string dungeonInfo = SafeFormat("{} in {}", quest->GetTitle(), player->GetMap()->GetMapName());
        eventChatter.DispatchGameEvent(player, EventKind::GuildDungeonComplete,
            { dungeonInfo, quest->GetQuestId() });
    }
}

ChatOnLearn::ChatOnLearn() : PlayerScript("ChatOnLearn") {}

void ChatOnLearn::OnPlayerLearnSpell(Player* player, uint32 spellID)
{
    if (!player)
        return;
    SpellInfo const* spellInfo = sSpellMgr->GetSpellInfo(spellID);
    std::string name = spellInfo ? spellInfo->SpellName[0] : std::to_string(spellID);
    eventChatter.DispatchGameEvent(player, EventKind::LearnedSpell, { name, spellID });
}

ChatOnDuel::ChatOnDuel() : PlayerScript("ChatOnDuel") {}

void ChatOnDuel::OnPlayerDuelRequest(Player* target, Player* challenger)
{
    if (!challenger || !target)
        return;
    eventChatter.DispatchGameEvent(challenger, EventKind::RequestedDuel, { target->GetName(), 0 });
}

void ChatOnDuel::OnPlayerDuelStart(Player* player1, Player* player2)
{
    if (!player1 || !player2)
        return;
    eventChatter.DispatchGameEvent(player1, EventKind::StartedDueling, { player2->GetName(), 0 });
}

void ChatOnDuel::OnPlayerDuelEnd(Player* winner, Player* loser, DuelCompleteType /*type*/)
{
    if (!winner || !loser)
        return;
    eventChatter.DispatchGameEvent(winner, EventKind::WonDuel, { loser->GetName(), 0 });
}

ChatOnLevelUp::ChatOnLevelUp() : PlayerScript("ChatOnLevelUp") {}

void ChatOnLevelUp::OnPlayerLevelChanged(Player* player, uint8 /*oldLevel*/)
{
    if (!player)
        return;
    std::string levelStr = std::to_string(player->GetLevel());
    eventChatter.DispatchGameEvent(player, EventKind::LeveledUp, { levelStr, player->GetLevel() });
    if (player->GetGuild() && g_EnableGuildEventChatter)
        eventChatter.DispatchGameEvent(player, EventKind::GuildLevelUp, { levelStr, player->GetLevel() });
}

ChatOnAchievement::ChatOnAchievement() : PlayerScript("ChatOnAchievement") {}

void ChatOnAchievement::OnPlayerAchievementComplete(Player* player, AchievementEntry const* achievement)
{
    if (!player || !achievement)
        return;
    eventChatter.DispatchGameEvent(player, EventKind::Achievement, { achievement->name[0], achievement->ID });
    if (player->GetGuild() && g_EnableGuildEventChatter && !PlayerbotsMgr::instance().GetPlayerbotAI(player))
        eventChatter.DispatchGameEvent(player, EventKind::GuildAchievement, { achievement->name[0], achievement->ID });
}

ChatOnGameObjectUse::ChatOnGameObjectUse() : AllGameObjectScript("ChatOnGameObjectUse") {}

bool ChatOnGameObjectUse::CanGameObjectGossipHello(Player* player, GameObject* go)
{
    if (!player || !go || !go->GetGOInfo())
        return false;
    eventChatter.DispatchGameEvent(player, EventKind::UsedObject, { go->GetGOInfo()->name, go->GetEntry() });
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
    eventChatter.DispatchGameEvent(player, EventKind::GuildJoin, { guild->GetName(), 0 });
}

void ChatOnGuildMemberChange::OnRemoveMember(Guild* guild, Player* player, bool /*isDisbanding*/, bool /*isKicked*/)
{
    if (!player || !guild || !g_EnableGuildEventChatter)
        return;
    eventChatter.DispatchGameEvent(player, EventKind::GuildLeave, { guild->GetName(), 0 });
}

void ChatOnGuildMemberChange::OnEvent(Guild* guild, uint8 eventType, ObjectGuid::LowType playerGuid1,
    ObjectGuid::LowType /*playerGuid2*/, uint8 newRank)
{
    if (!guild || !g_EnableGuildEventChatter)
        return;
    Player* player = ObjectAccessor::FindPlayerByLowGUID(playerGuid1);
    if (!player)
        return;
    if (eventType == GUILD_EVENT_LOG_PROMOTE_PLAYER)
        eventChatter.DispatchGameEvent(player, EventKind::GuildPromotion, { std::to_string(newRank), newRank });
    else if (eventType == GUILD_EVENT_LOG_DEMOTE_PLAYER)
        eventChatter.DispatchGameEvent(player, EventKind::GuildDemotion, { std::to_string(newRank), newRank });
}

ChatOnGuildLogin::ChatOnGuildLogin() : PlayerScript("ChatOnGuildLogin") {}

void ChatOnGuildLogin::OnPlayerLogin(Player* player)
{
    if (!player || !player->GetGuild() || !g_EnableGuildEventChatter)
        return;
    if (PlayerbotsMgr::instance().GetPlayerbotAI(player))
        return;
    eventChatter.DispatchGameEvent(player, EventKind::GuildLogin, { player->GetGuild()->GetName(), 0 });
}
