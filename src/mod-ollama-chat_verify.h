#ifndef MOD_OLLAMA_CHAT_VERIFY_H
#define MOD_OLLAMA_CHAT_VERIFY_H

#include <string>
#include "mod-ollama-chat_prompt.h"
#include "mod-ollama-chat_handler.h"

struct VerifyResult
{
    bool pass = true;
    std::string feedback;
};

VerifyResult VerifyReply(ChatIntent const& intent, std::string const& triggerMsg,
    std::string const& reply, ChatChannelSourceLocal channel);

#endif
