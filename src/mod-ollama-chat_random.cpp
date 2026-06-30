#include "mod-ollama-chat_random.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_sentiment.h"
#include "Log.h"
#include "Player.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "ObjectAccessor.h"
#include "Chat.h"
#include "ChannelMgr.h"
#include "Channel.h"
#include "fmt/core.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat-utilities.h"
#include "GridNotifiersImpl.h"
#include "CellImpl.h"
#include "Map.h"
#include "GridNotifiers.h"
#include "Guild.h"
#include "Group.h"
#include <vector>
#include <initializer_list>
#include <thread>
#include <ctime>
#include "Item.h"
#include "Bag.h"
#include "SpellMgr.h"
#include "AiFactory.h"
#include "ObjectMgr.h"
#include "QuestDef.h"

OllamaBotRandomChatter::OllamaBotRandomChatter() : WorldScript("OllamaBotRandomChatter") {}

namespace
{
std::string PickRandomLine(std::vector<std::string> const& lines)
{
    if (lines.empty())
        return "";
    return lines.size() == 1 ? lines[0] : lines[urand(0, lines.size() - 1)];
}

std::string PickEnvironmentSeed(std::initializer_list<std::vector<std::string> const*> tiers)
{
    for (auto const* tier : tiers)
    {
        if (tier && !tier->empty())
            return PickRandomLine(*tier);
    }
    return "";
}

struct BotAudience
{
    bool nearChatter = false;
    bool sayInRange = false;
    bool generalInZone = false;
    bool guildHasRealPlayer = false;
    bool partyHasRealPlayer = false;
};

bool IsRealPlayer(Player* player)
{
    return player && player->IsInWorld() && !PlayerbotsMgr::instance().GetPlayerbotAI(player);
}

void UpdateBotAudience(BotAudience& audience, Player* bot, Player* player)
{
    if (!IsRealPlayer(player))
        return;
    float const dist = bot->GetDistance(player);
    if (dist <= g_RandomChatterRealPlayerDistance)
        audience.nearChatter = true;
    if (!g_DisableForSayYell && dist <= g_SayDistance)
        audience.sayInRange = true;
    if (!g_DisableForCustomChannels && player->GetTeamId() == bot->GetTeamId() && player->GetZoneId() == bot->GetZoneId())
        audience.generalInZone = true;
    if (Guild* guild = bot->GetGuild())
        if (player->GetGuild() && player->GetGuild()->GetId() == guild->GetId())
            audience.guildHasRealPlayer = true;
    if (Group* group = bot->GetGroup())
        if (player->GetGroup() && player->GetGroup()->GetGUID() == group->GetGUID())
            audience.partyHasRealPlayer = true;
}

bool BotEligibleForRandomChatter(Player* bot, std::vector<Player*> const& realPlayers)
{
    Guild* guild = bot->GetGuild();
    for (Player* player : realPlayers)
    {
        if (!IsRealPlayer(player))
            continue;
        if (guild && player->GetGuild() && player->GetGuild()->GetId() == guild->GetId())
            return true;
        if (bot->GetDistance(player) <= g_RandomChatterRealPlayerDistance)
            return true;
    }
    return false;
}

BotAudience ComputeBotAudience(Player* bot, std::vector<Player*> const& realPlayers)
{
    BotAudience audience;
    for (Player* player : realPlayers)
        UpdateBotAudience(audience, bot, player);
    return audience;
}

BotAudience RefreshBotAudience(Player* bot)
{
    BotAudience audience;
    for (auto const& itr : ObjectAccessor::GetPlayers())
        UpdateBotAudience(audience, bot, itr.second);
    return audience;
}

ChatChannelSourceLocal RandomChatterChannel(Player* bot)
{
    return bot && bot->GetGroup() ? SRC_PARTY_LOCAL : SRC_GENERAL_LOCAL;
}

PromptBundle BuildRandomChatterPromptBundle(Player* bot, std::string const& environmentInfo, RandomIntent intent)
{
    ChatChannelSourceLocal channel = RandomChatterChannel(bot);

    if (g_RandomChatterPromptTemplate.empty())
        return BuildRandomChatterPrompt(bot, environmentInfo, intent, channel);

    PromptBundle bundle;
    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (!botAI)
        return bundle;

    std::string personality = GetBotPersonality(bot);
    std::string personalityPrompt = GetPersonalityPromptAddition(personality);
    AreaTableEntry const* botCurrentArea = botAI->GetCurrentArea();
    AreaTableEntry const* botCurrentZone = botAI->GetCurrentZone();

    bundle.user = SafeFormat(
        g_RandomChatterPromptTemplate,
        fmt::arg("bot_name", bot->GetName()),
        fmt::arg("bot_level", bot->GetLevel()),
        fmt::arg("bot_class", botAI->GetChatHelper()->FormatClass(bot->getClass())),
        fmt::arg("bot_race", botAI->GetChatHelper()->FormatRace(bot->getRace())),
        fmt::arg("bot_gender", bot->getGender() == GENDER_MALE ? "Male" : "Female"),
        fmt::arg("bot_role", ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot))),
        fmt::arg("bot_faction", bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde"),
        fmt::arg("bot_area", botCurrentArea ? botAI->GetLocalizedAreaName(botCurrentArea) : "UnknownArea"),
        fmt::arg("bot_zone", botCurrentZone ? botAI->GetLocalizedAreaName(botCurrentZone) : "UnknownZone"),
        fmt::arg("bot_map", bot->GetMap() ? bot->GetMap()->GetMapName() : "UnknownMap"),
        fmt::arg("bot_personality", personalityPrompt),
        fmt::arg("bot_personality_name", personality),
        fmt::arg("environment_info", environmentInfo));

    if (environmentInfo.empty())
    {
        std::vector<std::string> const* pool = nullptr;
        if (!g_RandomChatterPromptVariations.empty() && !g_RandomChatterQuestionVariations.empty())
            pool = urand(0, 1) == 0 ? &g_RandomChatterPromptVariations : &g_RandomChatterQuestionVariations;
        else if (!g_RandomChatterPromptVariations.empty())
            pool = &g_RandomChatterPromptVariations;
        else if (!g_RandomChatterQuestionVariations.empty())
            pool = &g_RandomChatterQuestionVariations;

        if (pool)
        {
            std::string variation = PickRandomLine(*pool);
            if (!variation.empty())
                bundle.user += " " + variation;
        }
    }

    bundle.user += " " + RandomIntentTaskLine(intent);

    BotContext ctx = OllamaPromptComposer::GatherBotContext(bot, nullptr);
    EnrichPromptBundle(bundle, bot, ctx, nullptr, channel);
    return bundle;
}
} // namespace

std::unordered_map<uint64_t, time_t> nextRandomChatTime;

void OllamaBotRandomChatter::OnUpdate(uint32 diff)
{
    if (!g_Enable)
        return;

    if (g_ConversationHistorySaveInterval > 0)
    {
        time_t now = time(nullptr);
        if (difftime(now, g_LastHistorySaveTime) >= g_ConversationHistorySaveInterval * 60)
        {
            SaveBotConversationHistoryToDB();
            g_LastHistorySaveTime = now;
        }
    }

    // Save sentiment data periodically
    if (g_EnableSentimentTracking && g_SentimentSaveInterval > 0)
    {
        time_t now = time(nullptr);
        if (difftime(now, g_LastSentimentSaveTime) >= g_SentimentSaveInterval * 60)
        {
            SaveBotPlayerSentimentsToDB();
            PurgeOrphanedSentiments();
            g_LastSentimentSaveTime = now;
        }
    }

    if (!g_EnableRandomChatter)
        return;

    static uint32_t timer = 0;
    if (timer <= diff)
    {
        timer = 30000;
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] RandomChatter tick fired (HandleRandomChatter called)");
        HandleRandomChatter();
    }
    else
    {
        timer -= diff;
    }
}

void OllamaBotRandomChatter::HandleRandomChatter()
{
    auto const& allPlayers = ObjectAccessor::GetPlayers();

    std::vector<Player*> realPlayers;
    for (auto const& itr : allPlayers)
    {
        Player* player = itr.second;
        if (!player->IsInWorld()) continue;
        if (!PlayerbotsMgr::instance().GetPlayerbotAI(player))
            realPlayers.push_back(player);
    }

    std::unordered_set<uint64_t> processedBotsThisTick;
    uint32_t botsQueuedThisTick = 0;

    for (auto const& itr : allPlayers)
    {
        Player* bot = itr.second;
        PlayerbotAI* ai = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
        if (!ai) continue;
        if (!bot->IsInWorld() || bot->IsBeingTeleported()) continue;
        if (processedBotsThisTick.count(bot->GetGUID().GetRawValue())) continue;

        if (!BotEligibleForRandomChatter(bot, realPlayers))
            continue;

        uint64_t guid = bot->GetGUID().GetRawValue();
        processedBotsThisTick.insert(guid);

        time_t now = time(nullptr);
        if (nextRandomChatTime.find(guid) == nextRandomChatTime.end())
        {
            nextRandomChatTime[guid] = now + urand(g_MinRandomInterval, g_MaxRandomInterval);
            continue;
        }
        if (now < nextRandomChatTime[guid])
            continue;
        if (urand(0, 99) >= g_RandomChatterBotCommentChance)
            continue;
        if (g_DisableRepliesInCombat && bot->IsInCombat())
            continue;
        if (botsQueuedThisTick >= g_RandomChatterMaxBotsPerPlayer)
            continue;

        BotAudience audience = ComputeBotAudience(bot, realPlayers);

        std::string environmentInfo;
        std::vector<std::string> tier1;
        std::vector<std::string> tier2;
        std::vector<std::string> tier3;
        std::vector<std::string> tier4;
        std::vector<std::string> guildComments;
        std::string guildPick;

        // Creature
        {
                Unit* unitInRange = nullptr;
                Acore::AnyUnitInObjectRangeCheck creatureCheck(bot, g_SayDistance);
                Acore::UnitSearcher<Acore::AnyUnitInObjectRangeCheck> creatureSearcher(bot, unitInRange, creatureCheck);
                Cell::VisitObjects(bot, creatureSearcher, g_SayDistance);
                if (unitInRange && unitInRange->GetTypeId() == TYPEID_UNIT)
                    if (!g_EnvCommentCreature.empty()) {
                        uint32_t idx = g_EnvCommentCreature.size() == 1 ? 0 : urand(0, g_EnvCommentCreature.size() - 1);
                        std::string templ = g_EnvCommentCreature[idx];
                        tier1.push_back(SafeFormat(templ, fmt::arg("creature_name", unitInRange->ToCreature()->GetName())));
                    }
            }

            // Game Object
            {
                Acore::GameObjectInRangeCheck goCheck(bot->GetPositionX(), bot->GetPositionY(), bot->GetPositionZ(), g_SayDistance);
                GameObject* goInRange = nullptr;
                Acore::GameObjectSearcher<Acore::GameObjectInRangeCheck> goSearcher(bot, goInRange, goCheck);
                Cell::VisitObjects(bot, goSearcher, g_SayDistance);
                if (goInRange)
                {
                    if (!g_EnvCommentGameObject.empty()) {
                        uint32_t idx = g_EnvCommentGameObject.size() == 1 ? 0 : urand(0, g_EnvCommentGameObject.size() - 1);
                        std::string templ = g_EnvCommentGameObject[idx];
                        std::string gameObjectName = goInRange->GetName();
                        tier1.push_back(SafeFormat(templ, fmt::arg("object_name", gameObjectName)));
                    }
                }
            }

            // Equipped Item
            {
                std::vector<Item*> equippedItems;
                for (uint8_t slot = EQUIPMENT_SLOT_START; slot < EQUIPMENT_SLOT_END; ++slot)
                    if (Item* item = bot->GetItemByPos(slot))
                        equippedItems.push_back(item);

                if (!equippedItems.empty())
                {
                    uint32_t eqIdx = equippedItems.size() == 1 ? 0 : urand(0, equippedItems.size() - 1);
                    Item* randomEquipped = equippedItems[eqIdx];
                    if (!g_EnvCommentEquippedItem.empty()) {
                        uint32_t tempIdx = g_EnvCommentEquippedItem.size() == 1 ? 0 : urand(0, g_EnvCommentEquippedItem.size() - 1);
                        std::string templ = g_EnvCommentEquippedItem[tempIdx];
                        tier2.push_back(SafeFormat(templ, fmt::arg("item_name", randomEquipped->GetTemplate()->Name1)));
                    }
                }
            }

            // Spell
            {
                struct NamedSpell
                {
                    uint32 id;
                    std::string name;
                    std::string effect;
                    std::string cost;
                };
                std::vector<NamedSpell> validSpells;
                for (const auto& spellPair : bot->GetSpellMap())
                {
                    uint32 spellId = spellPair.first;
                    const SpellInfo* spellInfo = sSpellMgr->GetSpellInfo(spellId);
                    if (!spellInfo) continue;
                    if (spellInfo->Attributes & SPELL_ATTR0_PASSIVE)
                        continue;
                    if (spellInfo->SpellFamilyName == SPELLFAMILY_GENERIC)
                        continue;
                    if (bot->HasSpellCooldown(spellId))
                        continue;

                    std::string effectText;
                    for (int i = 0; i < MAX_SPELL_EFFECTS; ++i)
                    {
                        if (!spellInfo->Effects[i].IsEffect())
                            continue;
                        switch (spellInfo->Effects[i].Effect)
                        {
                            case SPELL_EFFECT_SCHOOL_DAMAGE: effectText = "Deals damage"; break;
                            case SPELL_EFFECT_HEAL: effectText = "Heals the target"; break;
                            case SPELL_EFFECT_APPLY_AURA: effectText = "Applies an effect"; break;
                            case SPELL_EFFECT_DISPEL: effectText = "Dispels magic"; break;
                            case SPELL_EFFECT_THREAT: effectText = "Generates threat"; break;
                            default: continue;
                        }
                        if (!effectText.empty())
                            break;
                    }
                    if (effectText.empty())
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

                    validSpells.push_back({spellId, name, effectText, costText});
                }

                if (!validSpells.empty())
                {
                    uint32_t spellIdx = validSpells.size() == 1 ? 0 : urand(0, validSpells.size() - 1);
                    const NamedSpell& randomSpell = validSpells[spellIdx];
                    tier4.push_back(fmt::format("Might use '{}' here — {}.", randomSpell.name, randomSpell.effect));
                }
            }

            // Quest area — bot quest log or current zone (not full DB scan)
            if (!g_EnvCommentQuestArea.empty())
            {
                std::vector<std::string> questAreas;
                for (auto const& qs : bot->getQuestStatusMap())
                {
                    if (qs.second.Status != QUEST_STATUS_INCOMPLETE)
                        continue;
                    Quest const* qt = sObjectMgr->GetQuestTemplate(qs.first);
                    if (!qt)
                        continue;
                    uint32 zone = qt->GetZoneOrSort();
                    if (!zone)
                        continue;
                    if (AreaTableEntry const* area = sAreaTableStore.LookupEntry(zone))
                    {
                        uint32_t idx = g_EnvCommentQuestArea.size() == 1 ? 0 : urand(0, g_EnvCommentQuestArea.size() - 1);
                        questAreas.push_back(SafeFormat(g_EnvCommentQuestArea[idx],
                            fmt::arg("quest_area", area->area_name[LocaleConstant::LOCALE_enUS])));
                    }
                }
                if (questAreas.empty())
                {
                    if (AreaTableEntry const* area = sAreaTableStore.LookupEntry(bot->GetZoneId()))
                    {
                        uint32_t idx = g_EnvCommentQuestArea.size() == 1 ? 0 : urand(0, g_EnvCommentQuestArea.size() - 1);
                        questAreas.push_back(SafeFormat(g_EnvCommentQuestArea[idx],
                            fmt::arg("quest_area", area->area_name[LocaleConstant::LOCALE_enUS])));
                    }
                }
                if (!questAreas.empty())
                    tier3.push_back(PickRandomLine(questAreas));
            }

            // Vendor
            {
                Unit* unit = nullptr;
                Acore::AnyUnitInObjectRangeCheck check(bot, g_SayDistance);
                Acore::UnitSearcher<Acore::AnyUnitInObjectRangeCheck> searcher(bot, unit, check);
                Cell::VisitObjects(bot, searcher, g_SayDistance);

                if (unit && unit->GetTypeId() == TYPEID_UNIT)
                {
                    Creature* vendor = unit->ToCreature();
                    if (vendor->HasNpcFlag(UNIT_NPC_FLAG_VENDOR))
                    {
                        if (!g_EnvCommentVendor.empty()) {
                            uint32_t idx = g_EnvCommentVendor.size() == 1 ? 0 : urand(0, g_EnvCommentVendor.size() - 1);
                            std::string templ = g_EnvCommentVendor[idx];
                            tier1.push_back(SafeFormat(templ, fmt::arg("vendor_name", vendor->GetName())));
                        }
                    }
                }
            }

            // Questgiver
            {
                Unit* unit = nullptr;
                Acore::AnyUnitInObjectRangeCheck check(bot, g_SayDistance);
                Acore::UnitSearcher<Acore::AnyUnitInObjectRangeCheck> searcher(bot, unit, check);
                Cell::VisitObjects(bot, searcher, g_SayDistance);

                if (unit && unit->GetTypeId() == TYPEID_UNIT)
                {
                    Creature* giver = unit->ToCreature();
                    if (giver->HasNpcFlag(UNIT_NPC_FLAG_QUESTGIVER))
                    {
                        auto bounds = sObjectMgr->GetCreatureQuestRelationBounds(giver->GetEntry());
                        int n       = std::distance(bounds.first, bounds.second);
                        if (!g_EnvCommentQuestgiver.empty()) {
                            uint32_t idx = g_EnvCommentQuestgiver.size() == 1 ? 0 : urand(0, g_EnvCommentQuestgiver.size() - 1);
                            std::string templ = g_EnvCommentQuestgiver[idx];
                            tier1.push_back(SafeFormat(templ,
                                fmt::arg("questgiver_name", giver->GetName()),
                                fmt::arg("quest_count", n)
                            ));
                        }
                    }
                }
            }

            // Free bag slots
            {
                int freeSlots = 0;
                for (uint8 i = INVENTORY_SLOT_ITEM_START; i < INVENTORY_SLOT_ITEM_END; ++i)
                    if (!bot->GetItemByPos(i))
                        ++freeSlots;
                for (uint8 b = INVENTORY_SLOT_BAG_START; b < INVENTORY_SLOT_BAG_END; ++b)
                    if (Bag* bag = bot->GetBagByPos(b))
                        freeSlots += bag->GetFreeSlots();

                if (!g_EnvCommentBagSlots.empty()) {
                    uint32_t idx = g_EnvCommentBagSlots.size() == 1 ? 0 : urand(0, g_EnvCommentBagSlots.size() - 1);
                    std::string templ = g_EnvCommentBagSlots[idx];
                    tier2.push_back(SafeFormat(templ, fmt::arg("bag_slots", freeSlots)));
                }
            }

            // Dungeon
            {
                if (bot->GetMap() && bot->GetMap()->IsDungeon())
                {
                    std::string name = bot->GetMap()->GetMapName();
                    if (!g_EnvCommentDungeon.empty()) {
                        uint32_t idx = g_EnvCommentDungeon.size() == 1 ? 0 : urand(0, g_EnvCommentDungeon.size() - 1);
                        std::string templ = g_EnvCommentDungeon[idx];
                        tier1.push_back(SafeFormat(templ, fmt::arg("dungeon_name", name)));
                    }
                }
            }

            // Unfinished Quest
            {
                std::vector<std::string> unfinished;
                for (auto const& qs : bot->getQuestStatusMap())
                {
                    if (qs.second.Status == QUEST_STATUS_INCOMPLETE)
                    {
                        if (auto* qt = sObjectMgr->GetQuestTemplate(qs.first))
                            if (!g_EnvCommentUnfinishedQuest.empty()) {
                                uint32_t idx = g_EnvCommentUnfinishedQuest.size() == 1 ? 0 : urand(0, g_EnvCommentUnfinishedQuest.size() - 1);
                                std::string templ = g_EnvCommentUnfinishedQuest[idx];
                                unfinished.push_back(SafeFormat(templ, fmt::arg("quest_name", qt->GetTitle())));
                            }
                    }
                }
                if (!unfinished.empty())
                {
                    uint32_t uIdx = unfinished.size() == 1 ? 0 : urand(0, unfinished.size() - 1);
                    tier1.push_back(unfinished[uIdx]);
                }
            }

            // Guild-specific environment comments (if bot is in a guild with real players)
            if (g_EnableGuildRandomAmbientChatter && bot->GetGuild() && audience.guildHasRealPlayer
                && urand(0, 99) < g_GuildRandomChatterChance)
            {
                Guild* guild = bot->GetGuild();
                    // Guild member comments
                    if (!g_GuildEnvCommentGuildMember.empty())
                    {
                        std::string memberName = bot->GetName();
                        if (!memberName.empty())
                        {
                            uint32_t idx = urand(0, g_GuildEnvCommentGuildMember.size() - 1);
                            std::string templ = g_GuildEnvCommentGuildMember[idx];
                            guildComments.push_back(SafeFormat(templ, fmt::arg("member_name", memberName)));
                        }
                    }
                    // Guild MOTD comments
                    if (!g_GuildEnvCommentGuildMOTD.empty() && !guild->GetMOTD().empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildMOTD.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildMOTD[idx];
                        guildComments.push_back(SafeFormat(templ, fmt::arg("guild_motd", guild->GetMOTD())));
                    }
                    // Guild bank comments
                    if (!g_GuildEnvCommentGuildBank.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildBank.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildBank[idx];
                        guildComments.push_back(SafeFormat(templ, 
                            fmt::arg("bank_gold", guild->GetTotalBankMoney() / 10000)));
                    }
                    // Guild raid comments
                    if (bot->GetLevel() >= 58 && !g_GuildEnvCommentGuildRaid.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildRaid.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildRaid[idx];
                        guildComments.push_back(templ);
                    }
                    // Guild endgame comments
                    if (bot->GetLevel() >= 58 && !g_GuildEnvCommentGuildEndgame.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildEndgame.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildEndgame[idx];
                        guildComments.push_back(templ);
                    }
                    // Guild strategy comments
                    if (bot->GetLevel() >= 58 && !g_GuildEnvCommentGuildStrategy.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildStrategy.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildStrategy[idx];
                        guildComments.push_back(templ);
                    }
                    // Guild group/quest/grind comments
                    if (!g_GuildEnvCommentGuildGroup.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildGroup.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildGroup[idx];
                        guildComments.push_back(templ);
                    }
                    // Guild PvP comments
                    if (!g_GuildEnvCommentGuildPvP.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildPvP.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildPvP[idx];
                        guildComments.push_back(templ);
                    }
                    // Guild community/social comments
                    if (!g_GuildEnvCommentGuildCommunity.empty())
                    {
                        uint32_t idx = urand(0, g_GuildEnvCommentGuildCommunity.size() - 1);
                        std::string templ = g_GuildEnvCommentGuildCommunity[idx];
                        guildComments.push_back(templ);
                    }
            guildPick = PickRandomLine(guildComments);
            if (!guildPick.empty())
                tier1.push_back(guildPick);
        }

        environmentInfo = PickEnvironmentSeed({&tier1, &tier2, &tier3, &tier4});
        bool isGuildComment = !guildPick.empty() && environmentInfo == guildPick;

        RandomIntent intent = RandomIntent::ObserveZone;
        if (!environmentInfo.empty())
            intent = static_cast<RandomIntent>(urand(0, 3));
        else if (urand(0, 99) < 25)
            intent = RandomIntent::AskGroup;

        bool hasValidDestination = false;
        if (isGuildComment && bot->GetGuild())
            hasValidDestination = !g_DisableForGuild && audience.guildHasRealPlayer;
        else if (bot->GetGroup())
            hasValidDestination = !g_DisableForParty && audience.partyHasRealPlayer;
        else
            hasValidDestination = (!g_DisableForSayYell && audience.sayInRange)
                || (!g_DisableForCustomChannels && audience.generalInZone);

        if (!hasValidDestination)
        {
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping random chatter (no real player can hear the message)", bot->GetName());
            nextRandomChatTime[guid] = now + urand(g_MinRandomInterval, g_MaxRandomInterval);
            continue;
        }

        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] Random chatter queued for bot {}", bot->GetName());

        ++botsQueuedThisTick;
        uint64_t botGuid = bot->GetGUID().GetRawValue();
        std::string envCopy = environmentInfo;

        std::thread([botGuid, envCopy, intent, isGuildComment]() {
                try {
                    Player* botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                    if (!botPtr) return;

                    PromptBundle promptBundle = BuildRandomChatterPromptBundle(botPtr, envCopy, intent);
                    if (g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Random Message Prompt user: {}", promptBundle.user);
                    }

                    auto responseFuture = SubmitQuery(std::move(promptBundle));
                    if (!responseFuture.valid())
                        return;
                    std::string response = responseFuture.get();
                    if (response.empty())
                    {
                        if (g_DebugEnabled)
                            LOG_INFO("server.loading", "[OllamaChat] Bot skipped random chatter due to API error");
                        return;
                    }
                    
                    botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                    if (!botPtr) return;
                    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                    if (!botAI) return;
                    
                    // Simulate typing delay if enabled
                    if (g_EnableTypingSimulation)
                    {
                        uint32_t delay = g_TypingSimulationBaseDelay + (response.length() * g_TypingSimulationDelayPerChar);
                        if (g_DebugEnabled)
                            LOG_INFO("server.loading", "[OllamaChat] Bot simulating typing delay: {}ms for {} characters", 
                                     delay, response.length());
                        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                        
                        // Reacquire pointers after delay
                        botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                        if (!botPtr) return;
                        botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                        if (!botAI) return;
                    }
                    
                    BotAudience sendAudience = RefreshBotAudience(botPtr);

                    if (isGuildComment && botPtr->GetGuild())
                    {
                        if (g_DisableForGuild)
                        {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Guild random chatter skipped (guild channels disabled)");
                            return;
                        }
                        if (sendAudience.guildHasRealPlayer)
                        {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Bot Guild-Based Random Chatter: {}", response);
                            botAI->SayToGuild(response);
                            ProcessBotChatMessage(botPtr, response, SRC_GUILD_LOCAL, nullptr);
                        }
                        else if (g_DebugEnabled)
                        {
                            LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping guild random chatter (no real players in guild anymore)", botPtr->GetName());
                        }
                    }
                    else if (botPtr->GetGroup())
                    {
                        if (g_DisableForParty)
                        {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Party random chatter skipped (party channels disabled)");
                            return;
                        }
                        if (!sendAudience.partyHasRealPlayer)
                        {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping party random chatter (no real player in party)", botPtr->GetName());
                            return;
                        }
                        if (g_DebugEnabled)
                            LOG_INFO("server.loading", "[Ollama Chat] Bot Random Chatter Party: {}", response);
                        botAI->SayToParty(response);
                        ProcessBotChatMessage(botPtr, response, SRC_PARTY_LOCAL, nullptr);
                    }
                    else
                    {
                        std::vector<std::string> channels;
                        if (!g_DisableForCustomChannels && sendAudience.generalInZone)
                            channels.push_back("General");
                        if (!g_DisableForSayYell && sendAudience.sayInRange)
                            channels.push_back("Say");
                        if (channels.empty())
                        {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping random chatter (no audience)", botPtr->GetName());
                            return;
                        }

                        std::string selectedChannel = PickRandomLine(channels);
                        
                        if (selectedChannel == "Say") {
                            if (g_DebugEnabled)
                                LOG_INFO("server.loading", "[Ollama Chat] Bot {} Random Chatter Say (real player within {} yards): {}", botPtr->GetName(), g_SayDistance, response);
                            botAI->Say(response);
                            ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                        } else if (selectedChannel == "General") {
                            EnsureBotInGeneralChannel(botPtr);
                            Channel* generalChannel = nullptr;
                            if (ChannelMgr* cMgr = ChannelMgr::forTeam(botPtr->GetTeamId()))
                                generalChannel = cMgr->GetChannel("General", botPtr);

                            if (generalChannel)
                            {
                                TrySendGeneralChat(botPtr, response, "", generalChannel);
                                return;
                            }

                            if (sendAudience.sayInRange)
                            {
                                botAI->Say(response);
                                ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("server.loading", "[Ollama Chat] Exception in random chatter thread: {}", e.what());
                } catch (...) {
                    LOG_ERROR("server.loading", "[Ollama Chat] Unknown exception in random chatter thread");
                }
        }).detach();

        nextRandomChatTime[guid] = now + urand(g_MinRandomInterval, g_MaxRandomInterval);
    }
}
