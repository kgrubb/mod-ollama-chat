#ifndef MOD_OLLAMA_CHAT_API_H
#define MOD_OLLAMA_CHAT_API_H

#include <string>
#include <future>
#include "mod-ollama-chat_querymanager.h"
#include "mod-ollama-chat_prompt.h"

std::string QueryOllamaAPI(std::string const& prompt);
std::string QueryOllamaAPI(PromptBundle const& bundle);

bool IsValidAPIResponse(std::string const& response);

bool ValidateOllamaModel();

std::future<std::string> SubmitQuery(std::string const& prompt);
std::future<std::string> SubmitQuery(PromptBundle bundle);

extern QueryManager g_queryManager;

#endif
