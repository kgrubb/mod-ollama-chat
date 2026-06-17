#ifndef MOD_OLLAMA_CHAT_SANITIZE_H
#define MOD_OLLAMA_CHAT_SANITIZE_H

#include <string>

bool FinalizeGeneralChatLine(std::string& line, std::string const& triggerMsg);

#endif
