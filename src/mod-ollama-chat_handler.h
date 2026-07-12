#ifndef MOD_OLLAMA_CHAT_HANDLER_H
#define MOD_OLLAMA_CHAT_HANDLER_H

#include "ScriptMgr.h"
#include <string>
#include <vector>

class Player;
class Channel;
class PlayerbotAI;

enum ChatChannelSourceLocal
{
    SRC_UNDEFINED_LOCAL  = 0,
    SRC_SAY_LOCAL        = 1,
    SRC_PARTY_LOCAL      = 2,
    SRC_RAID_LOCAL       = 3,
    SRC_GUILD_LOCAL      = 4,
    SRC_OFFICER_LOCAL    = 5,
    SRC_YELL_LOCAL       = 6,
    SRC_WHISPER_LOCAL    = 7,
    SRC_GENERAL_LOCAL    = 17
};

extern const char* ChatChannelSourceLocalStr[];

struct GroupContext
{
    uint32_t memberCount = 0;
    uint32_t playerCount = 0;
    uint32_t botCount = 0;
    std::string statusLine;
    std::string partySection;
    std::vector<std::string> memberNames;
};

std::string rtrim(const std::string& s);
ChatChannelSourceLocal GetChannelSourceLocal(uint32_t type);
GroupContext BuildGroupContext(Player* bot);

std::string GetBotHistoryPrompt(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage);
std::vector<std::string> ChatHandler_GetVisibleLocations(Player* bot, float radius = 40.0f);
std::vector<std::string> ChatHandler_GetVisiblePlayers(Player* bot, float radius = 40.0f);
std::string BuildBotPromptContext(Player* bot);
std::string GenerateBotGameStateSnapshot(Player* bot, bool omitGroup = false);
std::string GenerateBotPrompt(Player* bot, std::string const& playerMessage, Player* player);
void ProcessBotChatMessage(Player* bot, const std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel);
void EnsureBotInGeneralChannel(Player* bot);
void EnsureBotInCityChannels(Player* bot);

void SaveBotConversationHistoryToDB();
void AppendBotConversation(uint64_t botGuid, uint64_t playerGuid, const std::string& playerMessage, const std::string& botReply, bool isEvent = false, ChatChannelSourceLocal channel = SRC_UNDEFINED_LOCAL, bool senderIsBot = false, bool verified = true);

void AppendZoneGeneralTranscript(uint32_t zoneId, std::string const& speaker, std::string const& text, bool isBot);
std::string FormatRecentGeneralTranscript(uint32_t zoneId, size_t maxLines = 12);
void MarkHumanGeneralActivity(uint32_t zoneId);
bool HumanActiveInZoneGeneralRecently(uint32_t zoneId);

bool TrySendGeneralChat(Player* bot, std::string& response, std::string const& triggerMsg, Channel* channel);

class PlayerBotChatHandler : public PlayerScript
{
public:
    PlayerBotChatHandler() : PlayerScript("PlayerBotChatHandler", {
        PLAYERHOOK_CAN_PLAYER_USE_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_PRIVATE_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_GROUP_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_GUILD_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_CHANNEL_CHAT,
        PLAYERHOOK_ON_LOGIN,
        PLAYERHOOK_ON_LOGOUT,
        PLAYERHOOK_ON_UPDATE_ZONE,
    }) {}
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Player* receiver) override;
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg) override;
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Group* group) override;
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Guild* guild) override;
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Channel* channel) override;
    void OnPlayerLogin(Player* player) override;
    void OnPlayerLogout(Player* player) override;
    void OnPlayerUpdateZone(Player* player, uint32 newZone, uint32 newArea) override;

    static void ProcessChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel = nullptr, Player* receiver = nullptr);
};

#endif // MOD_OLLAMA_CHAT_HANDLER_H
