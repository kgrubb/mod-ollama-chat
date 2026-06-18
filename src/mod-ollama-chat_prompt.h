#ifndef MOD_OLLAMA_CHAT_PROMPT_H
#define MOD_OLLAMA_CHAT_PROMPT_H

#include <cstdint>
#include <string>
#include <vector>
#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_handler.h"

class Player;

enum class PromptScenario
{
    PlayerChat,
    RandomChatter,
    EventReaction,
    SentimentAnalysis,
    MemoryMaintenance
};

enum class RandomIntent
{
    ObserveZone,
    ObserveEnvironment,
    SmallTalk,
    AskGroup
};

enum class KnowledgeLevel
{
    None,
    Hedge,
    Vague,
    Full
};

struct PromptBundle
{
    std::string system;
    std::string user;
    uint32_t maxTokens = 0;

    std::string CombinedForLegacyGenerate() const;
};

struct BotContext
{
    std::string botName;
    uint32_t botLevel = 0;
    std::string botClass;
    std::string botRace;
    std::string botGender;
    std::string botRole;
    std::string botFaction;
    std::string botGuild;
    std::string botArea;
    std::string botZone;
    std::string botMap;
    bool botInCombat = false;

    std::string personalityKey;
    std::string personalityLine;

    std::string playerName;
    uint32_t playerLevel = 0;
    std::string playerClass;
    std::string playerRole;
    std::string playerRace;
    float playerDistance = -1.0f;

    GroupContext groupCtx;
    uint8_t botRaceId = 0;
};

struct ChatIntent
{
    bool presenceQuestion = false;
    bool partyInvite = false;
    bool botReference = false;
    bool directQuestion = false;
    std::string referencedName;
};

struct ScenarioInput
{
    std::string playerMessage;
    std::string chatHistorySection;
    std::string memorySection;
    std::string sentimentSection;
    std::string knowledgeSection;
    KnowledgeLevel knowledgeLevel = KnowledgeLevel::None;
    std::string environmentSection;
    std::string contextSection;
    std::string nearbySection;
    std::string recentGeneralSection;
    std::string eventType;
    std::string eventDetail;
    std::string actorName;
    RandomIntent randomIntent = RandomIntent::ObserveZone;
    bool factualQuestion = false;
    ChatChannelSourceLocal chatChannel = SRC_UNDEFINED_LOCAL;
    std::string verificationFeedback;
    std::string intentTaskLines;
    ChatIntent chatIntent;

    std::string maintenanceFacts;
    std::string maintenanceNotes;
    std::string maintenanceTurns;
};

class OllamaPromptComposer
{
public:
    static void LoadPromptFiles();
    static BotContext GatherBotContext(Player* bot, Player* playerOrNull);
    static PromptBundle Build(PromptScenario scenario, BotContext const& ctx, ScenarioInput const& input);

    static bool IsFactualQuestion(std::string const& message);
    static bool IsSocialPresenceQuestion(std::string const& message);
    static std::string BuildRagQuery(std::string const& message, std::string const& zone, std::string const& area);
    static KnowledgeLevel RollKnowledgeLevel(BotContext const& ctx, std::vector<RAGResult> const& results);
    static std::string BuildKnowledgeSection(KnowledgeLevel level, std::string const& ragBullets);
};

std::string RandomIntentTaskLine(RandomIntent intent, ChatChannelSourceLocal channel = SRC_UNDEFINED_LOCAL);
std::string GetMemoryMaintenanceSystemPrompt();
std::string GetBotHistorySection(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage);
std::string GetBotCompactNearbySection(Player* bot);
ChatIntent DetectChatIntent(std::string const& message, std::vector<std::string> const& referenceNames);
std::string BuildIntentTaskLines(ChatIntent const& intent);
void EnrichPromptBundle(PromptBundle& bundle, Player* bot, BotContext const& ctx,
    Player* playerOrNull, ChatChannelSourceLocal channel, bool randomAmbient = false);

PromptBundle BuildPlayerChatPrompt(Player* bot, Player* player, std::string const& playerMessage,
    ChatChannelSourceLocal channel = SRC_UNDEFINED_LOCAL, ChatIntent const& intent = {},
    std::string const& verificationFeedback = {});
PromptBundle BuildRandomChatterPrompt(Player* bot, std::string const& environmentInfo, RandomIntent intent,
    ChatChannelSourceLocal channel = SRC_UNDEFINED_LOCAL);
PromptBundle BuildEventReactionPrompt(Player* bot, Player* actorPlayer, std::string const& eventType,
    std::string const& eventDetail, std::string const& actorName,
    ChatChannelSourceLocal channel = SRC_UNDEFINED_LOCAL);

#endif
