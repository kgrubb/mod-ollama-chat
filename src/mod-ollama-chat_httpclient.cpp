#include "mod-ollama-chat_httpclient.h"

#include <httplib.h>

#include "Log.h"
#include <chrono>
#include <regex>
#include <thread>

struct ParsedUrl
{
    std::string protocol;
    std::string host;
    int port = 0;
    std::string path;
    bool valid = false;
};

namespace
{
constexpr int kConnectTimeoutSec = 10;
constexpr int kReadTimeoutSec = 120;
constexpr int kMaxPostRetries = 3;
constexpr int kMaxGetRetries = 2;
constexpr int kRetryDelayMs = 1000;

ParsedUrl ParseUrl(std::string const& url)
{
    ParsedUrl parts;
    static std::regex const urlRegex(R"(^(https?)://([^:/]+)(?::(\d+))?(/.*)?$)");
    std::smatch match;

    if (!std::regex_match(url, match, urlRegex))
        return parts;

    parts.protocol = match[1].str();
    parts.host = match[2].str();
    if (match[3].matched)
    {
        try
        {
            parts.port = std::stoi(match[3].str());
        }
        catch (...)
        {
            return parts;
        }
    }
    else
    {
        parts.port = parts.protocol == "https" ? 443 : 11434;
    }

    parts.path = match[4].matched ? match[4].str() : "/";
    parts.valid = true;
    return parts;
}

httplib::Headers BuildHeaders(std::string const& bearerToken, std::string const& host)
{
    httplib::Headers headers = {
        {"Content-Type", "application/json"},
        {"User-Agent", "AzerothCore-OllamaChat/1.0"},
        {"Accept", "application/json"}
    };

    if (!bearerToken.empty())
        headers.emplace("Authorization", "Bearer " + bearerToken);

    if (host.find("ngrok") != std::string::npos || host.find("ngrok-free.app") != std::string::npos)
        headers.emplace("ngrok-skip-browser-warning", "true");

    return headers;
}

template<typename Fn>
httplib::Result DispatchRequest(ParsedUrl const& parts, int connectTimeout, int readTimeout, Fn&& fn)
{
    if (parts.protocol == "https")
    {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient client(parts.host, parts.port);
        client.enable_server_certificate_verification(false);
        client.set_connection_timeout(connectTimeout);
        client.set_read_timeout(readTimeout);
        client.set_write_timeout(readTimeout);
        return fn(client);
#else
        (void)parts;
        (void)connectTimeout;
        (void)readTimeout;
        (void)fn;
        return httplib::Result{};
#endif
    }

    httplib::Client client(parts.host, parts.port);
    client.set_connection_timeout(connectTimeout);
    client.set_read_timeout(readTimeout);
    client.set_write_timeout(readTimeout);
    return fn(client);
}

bool SslAvailable(ParsedUrl const& parts)
{
    if (parts.protocol != "https")
        return true;
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    LOG_ERROR("server.loading", "[Ollama Chat] HTTPS requested but SSL support not available.");
    return false;
#else
    return true;
#endif
}
} // namespace

OllamaHttpClient::OllamaHttpClient()
    : m_readTimeout(kReadTimeoutSec), m_connectTimeout(kConnectTimeoutSec), m_available(true)
{
}

OllamaHttpClient::~OllamaHttpClient() = default;

void OllamaHttpClient::MarkAvailable(bool available)
{
    bool const wasAvailable = m_available.exchange(available);
    if (available && !wasAvailable)
        LOG_INFO("server.loading", "[Ollama Chat] LLM API connection recovered");
    else if (!available && wasAvailable)
        LOG_ERROR("server.loading", "[Ollama Chat] LLM API unreachable");
}

OllamaHttpClient::HttpAttempt OllamaHttpClient::PostOnce(ParsedUrl const& parts, std::string const& jsonData, std::string const& bearerToken) const
{
    httplib::Headers const headers = BuildHeaders(bearerToken, parts.host);
    httplib::Result const response = DispatchRequest(parts, m_connectTimeout, m_readTimeout,
        [&](auto& client) { return client.Post(parts.path, headers, jsonData, "application/json"); });

    if (!response)
        return { {}, PostStatus::Retry };

    if (response->status == 200)
        return { response->body, PostStatus::Success };

    if (response->status == 408 || response->status == 429 || response->status >= 500)
        return { {}, PostStatus::Retry };

    return { response->body, PostStatus::Fail };
}

OllamaHttpClient::HttpAttempt OllamaHttpClient::GetOnce(ParsedUrl const& parts, std::string const& bearerToken) const
{
    httplib::Headers const headers = BuildHeaders(bearerToken, parts.host);
    httplib::Result const response = DispatchRequest(parts, m_connectTimeout, m_readTimeout,
        [&](auto& client) { return client.Get(parts.path, headers); });

    if (!response)
        return { {}, PostStatus::Retry };

    if (response->status == 200)
        return { response->body, PostStatus::Success };

    if (response->status == 408 || response->status == 429 || response->status >= 500)
        return { {}, PostStatus::Retry };

    return { response->body, PostStatus::Fail };
}

std::string OllamaHttpClient::Post(std::string const& url, std::string const& jsonData, std::string const& bearerToken)
{
    ParsedUrl const parts = ParseUrl(url);
    if (!parts.valid)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] Invalid URL format: {}", url);
        return "";
    }

    if (!SslAvailable(parts))
        return "";

    for (int attempt = 0; attempt < kMaxPostRetries; ++attempt)
    {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs * attempt));

        try
        {
            HttpAttempt const result = PostOnce(parts, jsonData, bearerToken);
            if (result.status == PostStatus::Success)
            {
                MarkAvailable(true);
                return result.body;
            }
            if (result.status == PostStatus::Fail)
            {
                MarkAvailable(true);
                return result.body;
            }
        }
        catch (std::exception const& e)
        {
            if (attempt == kMaxPostRetries - 1)
                LOG_ERROR("server.loading", "[Ollama Chat] HTTP client exception: {}", e.what());
        }
    }

    LOG_ERROR("server.loading", "[Ollama Chat] HTTP request failed after {} attempts to {}:{}{}",
        kMaxPostRetries, parts.host, parts.port, parts.path);
    MarkAvailable(false);
    return "";
}

std::string OllamaHttpClient::Get(std::string const& url, std::string const& bearerToken)
{
    ParsedUrl const parts = ParseUrl(url);
    if (!parts.valid)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] Invalid URL format: {}", url);
        return "";
    }

    if (!SslAvailable(parts))
        return "";

    for (int attempt = 0; attempt < kMaxGetRetries; ++attempt)
    {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs * attempt));

        try
        {
            HttpAttempt const result = GetOnce(parts, bearerToken);
            if (result.status == PostStatus::Success || result.status == PostStatus::Fail)
                return result.body;
        }
        catch (std::exception const& e)
        {
            if (attempt == kMaxGetRetries - 1)
                LOG_ERROR("server.loading", "[Ollama Chat] HTTP GET exception: {}", e.what());
        }
    }

    return "";
}

void OllamaHttpClient::SetTimeout(int seconds)
{
    m_readTimeout = seconds;
}
