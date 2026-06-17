#ifndef MOD_OLLAMA_CHAT_EVENTS_H
#define MOD_OLLAMA_CHAT_EVENTS_H

#include "ScriptMgr.h"
#include "Player.h"
#include <string>

class OllamaBotEventChatter
{
public:
    void DispatchGameEvent(Player* source, std::string type, std::string detail);
    void QueueEvent(Player* bot, std::string type, std::string detail, std::string actorName, bool isGuildEvent = false);
};

class ChatOnKill : public PlayerScript
{
public:
    ChatOnKill();
    void OnPlayerCreatureKill(Player* killer, Creature* victim) override;
    void OnPlayerPVPKill(Player* killer, Player* killed) override;
    void OnPlayerCreatureKilledByPet(Player* owner, Creature* victim) override;
};

class ChatOnLoot : public PlayerScript
{
public:
    ChatOnLoot();
    void OnPlayerStoreNewItem(Player* player, Item* item, uint32 count) override;
};

class ChatOnDeath : public PlayerScript
{
public:
    ChatOnDeath();
    void OnPlayerJustDied(Player* player) override;
};

class ChatOnQuest : public PlayerScript
{
public:
    ChatOnQuest();
    void OnPlayerCompleteQuest(Player* player, Quest const* quest) override;
};

class ChatOnLearn : public PlayerScript
{
public:
    ChatOnLearn();
    void OnPlayerLearnSpell(Player* player, uint32 spellID) override;
};

class ChatOnDuel : public PlayerScript
{
public:
    ChatOnDuel();
    void OnPlayerDuelRequest(Player* target, Player* challenger) override;
    void OnPlayerDuelStart(Player* player1, Player* player2) override;
    void OnPlayerDuelEnd(Player* winner, Player* loser, DuelCompleteType type) override;
};

class ChatOnLevelUp : public PlayerScript
{
public:
    ChatOnLevelUp();
    void OnPlayerLevelChanged(Player* player, uint8 oldLevel) override;
};

class ChatOnAchievement : public PlayerScript
{
public:
    ChatOnAchievement();
    void OnPlayerAchievementComplete(Player* player, AchievementEntry const* achievement) override;
};

class ChatOnGameObjectUse : public AllGameObjectScript
{
public:
    ChatOnGameObjectUse();
    bool CanGameObjectGossipHello(Player* player, GameObject* go) override;
};

class ChatOnGuildMemberChange : public GuildScript
{
public:
    ChatOnGuildMemberChange();
    void OnAddMember(Guild* guild, Player* player, uint8& plRank) override;
    void OnRemoveMember(Guild* guild, Player* player, bool isDisbanding, bool isKicked) override;
    void OnEvent(Guild* guild, uint8 eventType, ObjectGuid::LowType playerGuid1, ObjectGuid::LowType playerGuid2, uint8 newRank) override;
};

class ChatOnGuildLogin : public PlayerScript
{
public:
    ChatOnGuildLogin();
    void OnPlayerLogin(Player* player) override;
};

#endif // MOD_OLLAMA_CHAT_EVENTS_H
