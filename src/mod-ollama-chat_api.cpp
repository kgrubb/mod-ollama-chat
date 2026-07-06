#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_httpclient.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <thread>
#include <future>
#include <cctype>
#include <vector>

namespace
{
OllamaHttpClient& SharedHttpClient()
{
    static OllamaHttpClient client;
    return client;
}
}

static std::string StripWrappingQuotes(std::string const& response)
{
    if (response.size() >= 2 && response.front() == '"' && response.back() == '"')
        return response.substr(1, response.size() - 2);
    return response;
}

static bool IsOpenAIChatCompletionsUrl(std::string url)
{
    for (char& c : url)
        c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    return url.find("/v1/chat/completions") != std::string::npos || url.find("/chat/completions") != std::string::npos;
}

static std::string TruncateForLog(std::string const& text, size_t maxLen = 1024)
{
    if (text.size() <= maxLen)
        return text;
    return text.substr(0, maxLen) + "...";
}

static std::string SummarizeChatCompletionBody(std::string const& responseBuffer)
{
    if (responseBuffer.empty())
        return "";

    try
    {
        nlohmann::json j = nlohmann::json::parse(responseBuffer);
        j.erase("timings");

        nlohmann::json summary = nlohmann::json::object();
        if (j.contains("model"))
            summary["model"] = j["model"];
        if (j.contains("usage"))
            summary["usage"] = j["usage"];
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty())
        {
            nlohmann::json const& choice = j["choices"][0];
            nlohmann::json choiceSummary = nlohmann::json::object();
            if (choice.contains("finish_reason"))
                choiceSummary["finish_reason"] = choice["finish_reason"];
            if (choice.contains("message") && choice["message"].is_object())
            {
                nlohmann::json const& msg = choice["message"];
                if (msg.contains("role"))
                    choiceSummary["role"] = msg["role"];
                if (msg.contains("content") && msg["content"].is_string())
                    choiceSummary["content"] = msg["content"].get<std::string>();
                else if (msg.contains("content") && msg["content"].is_null())
                    choiceSummary["content"] = nullptr;
                if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string())
                {
                    std::string reasoning = msg["reasoning_content"].get<std::string>();
                    if (reasoning.size() > 120)
                        reasoning = reasoning.substr(0, 120) + "...";
                    choiceSummary["reasoning_content"] = reasoning;
                }
            }
            else if (choice.contains("text") && choice["text"].is_string())
                choiceSummary["text"] = choice["text"].get<std::string>();
            summary["choice"] = std::move(choiceSummary);
        }
        return summary.dump();
    }
    catch (...)
    {
        return TruncateForLog(responseBuffer);
    }
}

static std::string TrimReply(std::string s)
{
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

static std::string ExtractApiErrorMessage(std::string const& responseBuffer)
{
    if (responseBuffer.empty())
        return "";
    if (responseBuffer.find("\"error\"") == std::string::npos &&
        responseBuffer.find("\"detail\"") == std::string::npos)
        return "";

    try
    {
        nlohmann::json const root = nlohmann::json::parse(responseBuffer);
        if (root.contains("error"))
        {
            if (root["error"].is_string())
                return root["error"].get<std::string>();
            if (root["error"].is_object() && root["error"].contains("message") && root["error"]["message"].is_string())
                return root["error"]["message"].get<std::string>();
            return root["error"].dump();
        }
        if (root.contains("detail"))
        {
            if (root["detail"].is_string())
                return root["detail"].get<std::string>();
            return root["detail"].dump();
        }
    }
    catch (...)
    {
    }

    return "";
}

static std::string DeriveModelsUrl(std::string url)
{
    std::string lower = url;
    for (char& c : lower)
        c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    auto replaceSuffix = [&](std::string const& from, std::string const& to) -> bool
    {
        size_t const pos = lower.find(from);
        if (pos == std::string::npos)
            return false;
        url.replace(pos, from.size(), to);
        lower.replace(pos, from.size(), to);
        return true;
    };

    if (replaceSuffix("/v1/chat/completions", "/v1/models") ||
        replaceSuffix("/api/chat/completions", "/api/models") ||
        replaceSuffix("/chat/completions", "/models") ||
        replaceSuffix("/api/generate", "/api/tags"))
        return url;

    size_t const slash = url.rfind('/');
    if (slash != std::string::npos && slash > url.find("://") + 2)
        return url.substr(0, slash) + "/models";

    return url + "/models";
}

static bool ParseOpenAIResponse(std::string const& responseBuffer, std::string& botReply)
{
    nlohmann::json const root = nlohmann::json::parse(responseBuffer);
    if (root.contains("error") || root.contains("detail"))
    {
        LOG_ERROR("server.loading", "[OllamaChat] LLM API error: {}", ExtractApiErrorMessage(responseBuffer));
        return false;
    }
    if (root.contains("choices") && root["choices"].is_array() && !root["choices"].empty())
    {
        nlohmann::json const& choice = root["choices"][0];
        if (choice.contains("message") && choice["message"].is_object())
        {
            nlohmann::json const& message = choice["message"];
            if (message.contains("content") && message["content"].is_string())
                botReply = message["content"].get<std::string>();
            else if (message.contains("content") && message["content"].is_null())
                botReply.clear();
        }
        else if (choice.contains("text") && choice["text"].is_string())
            botReply = choice["text"].get<std::string>();
    }
    botReply = TrimReply(botReply);
    return true;
}

static std::string QueryOllamaAPIInternal(std::string const& systemText, std::string const& userPrompt, uint32_t maxTokensOverride = 0)
{
    OllamaHttpClient& httpClient = SharedHttpClient();

    std::string url   = g_OllamaUrl;
    std::string model = g_OllamaModel;

    std::string sanitizedPrompt = SanitizeUTF8(userPrompt);
    std::string effectiveSystem = !systemText.empty()
        ? systemText
        : (!g_SystemPromptOverride.empty() ? g_SystemPromptOverride : g_OllamaSystemPrompt);
    uint32_t const maxTokens = maxTokensOverride > 0 ? maxTokensOverride : g_OllamaNumPredict;

    if (IsOpenAIChatCompletionsUrl(url))
    {
        nlohmann::json messages = nlohmann::json::array();
        if (!effectiveSystem.empty())
            messages.push_back(nlohmann::json{{"role", "system"}, {"content", SanitizeUTF8(effectiveSystem)}});
        messages.push_back(nlohmann::json{{"role", "user"}, {"content", sanitizedPrompt}});

        nlohmann::json requestData = {
            {"model", model},
            {"messages", messages},
            {"stream", false}
        };

        if (maxTokens > 0)
            requestData["max_tokens"] = maxTokens;
        if (g_OllamaTemperature != 0.8f)
            requestData["temperature"] = g_OllamaTemperature;
        if (g_OllamaTopP != 0.95f)
            requestData["top_p"] = g_OllamaTopP;

        requestData["chat_template_kwargs"] = nlohmann::json{
            {"enable_thinking", g_ThinkModeEnableForModule}
        };

        if (!g_OllamaStop.empty()) {
            std::vector<std::string> stopSeqs;
            std::stringstream ss(g_OllamaStop);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t start = item.find_first_not_of(" \t");
                size_t end = item.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos)
                    stopSeqs.push_back(item.substr(start, end - start + 1));
            }
            if (!stopSeqs.empty())
                requestData["stop"] = stopSeqs;
        }

        std::string requestDataStr = requestData.dump();
        std::string responseBuffer = httpClient.Post(url, requestDataStr, g_OllamaApiKey);

        if (responseBuffer.empty())
        {
            LOG_ERROR("server.loading", "[OllamaChat] LLM API unreachable at {}", url);
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[OllamaChat] Debug: Empty response buffer from HTTP client. Model: {}", model);
            }
            return "";
        }

        auto parseResponse = [&](std::string const& buffer, std::string& outReply) -> bool
        {
            try
            {
                return ParseOpenAIResponse(buffer, outReply);
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("server.loading", "[OllamaChat] JSON parsing failed: {} — body: {}", e.what(),
                    SummarizeChatCompletionBody(buffer));
                if(g_DebugEnabled)
                    LOG_INFO("server.loading", "[OllamaChat] Debug: Response buffer content: {}", buffer);
                return false;
            }
        };

        std::string botReply;
        if (!parseResponse(responseBuffer, botReply))
            return "";

        if (botReply.empty())
        {
            LOG_ERROR("server.loading",
                "[OllamaChat] Empty response from LLM API (model: {}). Summary: {}",
                model, SummarizeChatCompletionBody(responseBuffer));
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[OllamaChat] Debug: Full API body: {}", responseBuffer);
            return "";
        }

        botReply = StripWrappingQuotes(botReply);

        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Parsed bot response: {}", botReply);
        }

        return botReply;
    }

    nlohmann::json requestData = {
        {"model",  model},
        {"prompt", sanitizedPrompt},
        {"stream", false}
    };

    // Create options object for model parameters
    nlohmann::json options;
    bool hasOptions = false;

    // Only include if set (do not send defaults if user did not set them)
    if (g_OllamaNumPredict > 0) {
        options["num_predict"] = g_OllamaNumPredict;
        hasOptions = true;
    }
    if (g_OllamaTemperature != 0.8f) {
        options["temperature"] = g_OllamaTemperature;
        hasOptions = true;
    }
    if (g_OllamaTopP != 0.95f) {
        options["top_p"] = g_OllamaTopP;
        hasOptions = true;
    }
    if (g_OllamaRepeatPenalty != 1.1f) {
        options["repeat_penalty"] = g_OllamaRepeatPenalty;
        hasOptions = true;
    }
    if (g_OllamaNumCtx > 0) {
        options["num_ctx"] = g_OllamaNumCtx;
        hasOptions = true;
    }
    if (g_OllamaNumThreads > 0) {
        options["num_thread"] = g_OllamaNumThreads;
        hasOptions = true;
        if(g_DebugEnabled) {
            //LOG_INFO("server.loading", "[Ollama Chat] Setting num_thread to: {}", g_OllamaNumThreads);
        }
    } else if(g_DebugEnabled) {
        //LOG_INFO("server.loading", "[Ollama Chat] g_OllamaNumThreads is: {} (not sending num_thread)", g_OllamaNumThreads);
    }
    if (!g_OllamaSeed.empty()) {
        try {
            int seedValue = std::stoi(g_OllamaSeed);
            options["seed"] = seedValue; 
            hasOptions = true;
        } catch (const std::exception& e) {
            if(g_DebugEnabled) {
                LOG_INFO("server.loading", "[Ollama Chat] Invalid seed value: {}", g_OllamaSeed);
            }
        }
    }

    // Add options object if any options were set
    if (hasOptions) {
        requestData["options"] = options;
    }

    // Root-level parameters (these stay at root level)
    if (!g_OllamaStop.empty()) {
        // If comma-separated, convert to array
        std::vector<std::string> stopSeqs;
        std::stringstream stopStream(g_OllamaStop);
        std::string item;
        while (std::getline(stopStream, item, ',')) {
            // trim whitespace
            size_t start = item.find_first_not_of(" \t");
            size_t end = item.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos)
                stopSeqs.push_back(item.substr(start, end - start + 1));
        }
        if (!stopSeqs.empty())
            requestData["stop"] = stopSeqs;
    }
    if (!effectiveSystem.empty())
        requestData["system"] = SanitizeUTF8(effectiveSystem);

    if (g_ThinkModeEnableForModule)
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] LLM set to Think mode.");
        }
        requestData["think"] = true;
        requestData["hidethinking"] = true;
    }

    std::string requestDataStr = requestData.dump();

    // Make HTTP POST request using our custom client
    std::string responseBuffer = httpClient.Post(url, requestDataStr, g_OllamaApiKey);

    if (responseBuffer.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] LLM API unreachable at {}", url);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Empty response buffer from HTTP client. Model: {}", model);
        }
        return "";
    }

    std::string const apiError = ExtractApiErrorMessage(responseBuffer);
    if (!apiError.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] LLM API error: {}", apiError);
        return "";
    }

    std::stringstream ss(responseBuffer);
    std::string line;
    std::ostringstream extractedResponse;

    try
    {
        while (std::getline(ss, line))
        {
            if (line.empty() || std::all_of(line.begin(), line.end(), [](unsigned char ch) { return std::isspace(ch); }))
                continue;

            nlohmann::json jsonResponse = nlohmann::json::parse(line);

            if (jsonResponse.contains("response") && !jsonResponse["response"].get<std::string>().empty())
            {
                extractedResponse << jsonResponse["response"].get<std::string>();
            }
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: JSON parsing failed. Exception: {}", e.what());
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Response buffer content: {}", responseBuffer);
        }
        return "";
    }

    std::string botReply = extractedResponse.str();

    botReply = StripWrappingQuotes(botReply);

    // Check for unclosed think tags
    if (botReply.find("<think>") != std::string::npos || botReply.find("</think>") != std::string::npos)
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: Unclosed <think> tags detected in response. This usually means the model's output was truncated.");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.ThinkModeEnableForModule = 1' in mod_ollama_chat.conf");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumPredict = 0' (unlimited tokens) in mod_ollama_chat.conf");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumCtx = 0' (model default context) in mod_ollama_chat.conf");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Partial response with think tags: {}", botReply);
        }
        return "";
    }

    if (botReply.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: Empty response extracted from API. Model may not have generated any output.");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Raw extracted response was empty.");
        }
        return "";
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Parsed bot response: {}", botReply);

        if (g_ThinkModeEnableForModule)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Bot used think.");
            }
        }
    }

    return botReply;
}

std::string QueryOllamaAPI(std::string const& prompt)
{
    return QueryOllamaAPIInternal("", prompt);
}

std::string QueryOllamaAPI(PromptBundle const& bundle)
{
    return QueryOllamaAPIInternal(bundle.system, bundle.user, bundle.maxTokens);
}

// Helper function to check if a response is valid (not empty and not an error)
bool IsValidAPIResponse(const std::string& response)
{
    return !response.empty();
}

bool ValidateOllamaModel()
{
    if (g_OllamaModel.empty() || g_OllamaUrl.empty())
        return false;

    std::string const modelsUrl = DeriveModelsUrl(g_OllamaUrl);
    std::string const body = SharedHttpClient().Get(modelsUrl, g_OllamaApiKey);
    if (body.empty())
    {
        LOG_WARN("server.loading", "[Ollama Chat] Model validation skipped — {} unreachable", modelsUrl);
        return true;
    }

    try
    {
        nlohmann::json const root = nlohmann::json::parse(body);
        std::vector<std::string> available;

        auto collect = [&](nlohmann::json const& arr, char const* key)
        {
            for (auto const& entry : arr)
            {
                std::string id;
                if (key && entry.contains(key) && entry[key].is_string())
                    id = entry[key].get<std::string>();
                else if (entry.is_string())
                    id = entry.get<std::string>();
                if (!id.empty())
                    available.push_back(id);
            }
        };

        if (root.contains("data") && root["data"].is_array())
            collect(root["data"], "id");
        if (root.contains("models") && root["models"].is_array())
            collect(root["models"], "name");

        for (std::string const& id : available)
        {
            if (id == g_OllamaModel)
                return true;
        }

        if (available.empty())
        {
            LOG_WARN("server.loading", "[Ollama Chat] Model validation inconclusive — no models at {}", modelsUrl);
            return true;
        }

        std::ostringstream listed;
        for (size_t i = 0; i < available.size() && i < 8; ++i)
        {
            if (i > 0)
                listed << ", ";
            listed << available[i];
        }
        if (available.size() > 8)
            listed << ", ...";

        LOG_ERROR("server.loading", "[Ollama Chat] Model '{}' not found. Available: {}", g_OllamaModel, listed.str());
        return false;
    }
    catch (std::exception const& e)
    {
        LOG_WARN("server.loading", "[Ollama Chat] Model validation skipped — bad response from {}: {}", modelsUrl, e.what());
        return true;
    }
}

QueryManager g_queryManager;

std::future<std::string> SubmitQuery(std::string const& prompt)
{
    return g_queryManager.submitQuery(prompt);
}

std::future<std::string> SubmitQuery(PromptBundle bundle)
{
    return g_queryManager.submitQuery(std::move(bundle));
}
