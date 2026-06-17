#ifndef OLLAMA_HTTP_CLIENT_H
#define OLLAMA_HTTP_CLIENT_H

#include <atomic>
#include <string>

struct ParsedUrl;

class OllamaHttpClient
{
public:
    OllamaHttpClient();
    ~OllamaHttpClient();

    std::string Post(std::string const& url, std::string const& jsonData, std::string const& bearerToken = "");
    std::string Get(std::string const& url, std::string const& bearerToken = "");
    void SetTimeout(int seconds);

private:
    enum class PostStatus { Success, Retry, Fail };

    struct HttpAttempt
    {
        std::string body;
        PostStatus status = PostStatus::Fail;
    };

    HttpAttempt PostOnce(ParsedUrl const& parts, std::string const& jsonData, std::string const& bearerToken) const;
    HttpAttempt GetOnce(ParsedUrl const& parts, std::string const& bearerToken) const;
    void MarkAvailable(bool available);

    int m_readTimeout;
    int m_connectTimeout;
    std::atomic<bool> m_available;
};

#endif // OLLAMA_HTTP_CLIENT_H
