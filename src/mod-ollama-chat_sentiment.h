#ifndef MOD_OLLAMA_CHAT_SENTIMENT_H
#define MOD_OLLAMA_CHAT_SENTIMENT_H

#include <string>
#include <cstdint>
#include "Player.h"

float GetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid);
void SetBotPlayerSentiment(uint64_t botGuid, uint64_t playerGuid, float sentimentValue);
void UpdateBotPlayerSentiment(Player* bot, Player* player, const std::string& message);
std::string GetSentimentPromptAddition(Player* bot, Player* player);
void LoadBotPlayerSentimentsFromDB();
void SaveBotPlayerSentimentsToDB();
void PurgeOrphanedSentiments();
void InitializeSentimentTracking();

#endif // MOD_OLLAMA_CHAT_SENTIMENT_H
