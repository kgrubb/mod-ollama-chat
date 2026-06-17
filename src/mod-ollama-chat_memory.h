#ifndef MOD_OLLAMA_CHAT_MEMORY_H
#define MOD_OLLAMA_CHAT_MEMORY_H

#include <string>
#include <cstdint>
#include <vector>
#include <utility>

namespace OllamaMemory {
inline constexpr uint32_t DequeCap = 12;
inline constexpr uint32_t SaveIntervalMinutes = 10;
inline constexpr uint32_t MaxSemanticChars = 1500;
inline constexpr uint32_t CompactionThreshold = 3;
inline constexpr uint32_t CompactionMaxAgeMinutes = 60;
inline constexpr uint32_t EpisodicMaxAgeDays = 30;
inline constexpr uint32_t RecallMaxItems = 5;
inline constexpr float RecallThreshold = 0.1f;
inline constexpr uint32_t MaxConcurrentCompactions = 1;
inline constexpr uint32_t CompactionsPerTick = 1;
inline constexpr uint32_t ArchiveRecallWindow = 30;
inline constexpr uint32_t ArchiveMaxRowsPerPair = 200;
inline constexpr uint32_t ArchivePendingCap = 48;
inline constexpr uint32_t NudgeMinIntervalSeconds = 300;
inline constexpr uint32_t CompactionFailCooldownSeconds = 900;
inline constexpr uint32_t CompactionAgeSweepIntervalSeconds = 300;
inline constexpr uint32_t MaxCompactionQueue = 32;
inline constexpr uint32_t MemoryTickIntervalMs = 3000;
inline constexpr uint32_t MemoryQueryMaxTokens = 256;
}

enum class MemoryCompactionEnqueueResult : uint8_t
{
    Enqueued,
    NothingToCompact,
    AlreadyPending
};

void InitializeBotMemory();
bool SaveBotMemoryToDB(bool blocking = false);
MemoryCompactionEnqueueResult EnqueueMemoryCompaction(uint64_t botGuid, uint64_t playerGuid, bool force = false);
void MaybeEnqueueMemoryCompaction(uint64_t botGuid, uint64_t playerGuid);
void MaybeEnqueueMemoryNudge(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage, bool isEvent);
void ProcessMemoryCompactionTick();
void AppendBotMemoryTurn(uint64_t botGuid, uint64_t playerGuid, std::string const& playerMessage,
    std::string const& botReply, bool isEvent, bool verified = true);
std::string GetMemoryPromptAddition(uint64_t botGuid, uint64_t playerGuid, const std::string& query, const std::string& playerName);
uint32_t GetPendingTurnCount(uint64_t botGuid, uint64_t playerGuid);
void ResetBotMemory(uint64_t botGuid, uint64_t playerGuid);
std::vector<std::pair<uint64_t, uint64_t>> CollectMemoryPairs(uint64_t botGuid, uint64_t playerGuid);
std::string GetStoredPlayerName(uint64_t botGuid, uint64_t playerGuid);
std::string GetMemoryDebugInfo(uint64_t botGuid, uint64_t playerGuid);

#endif
