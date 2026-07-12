#ifndef MOD_OLLAMA_CHAT_RANDOM_H
#define MOD_OLLAMA_CHAT_RANDOM_H

#include "ScriptMgr.h"
#include <cstdint>

class OllamaBotRandomChatter : public WorldScript
{
public:
    OllamaBotRandomChatter();
    void OnUpdate(uint32 diff) override;

private:
    void HandleRandomChatter();
};

void ClearBotRandomChatterState(uint64_t botGuid);

#endif // MOD_OLLAMA_CHAT_RANDOM_H
