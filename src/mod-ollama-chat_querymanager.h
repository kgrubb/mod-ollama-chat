#ifndef MOD_OLLAMA_CHAT_QUERYMANAGER_H
#define MOD_OLLAMA_CHAT_QUERYMANAGER_H

#include <string>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include "mod-ollama-chat_prompt.h"

class QueryManager {
public:
    QueryManager();
    void setMaxConcurrentQueries(int maxQueries);
    std::future<std::string> submitQuery(std::string const& prompt);
    std::future<std::string> submitQuery(PromptBundle bundle);
    bool ShouldDeferMemoryWork();

private:
    struct QueryTask {
        PromptBundle bundle;
        std::promise<std::string> promise;
    };

    void processQuery(PromptBundle bundle, std::promise<std::string> promise);

    int maxConcurrentQueries;
    int currentQueries;
    std::mutex mutex_;
    std::queue<QueryTask> taskQueue;
};

#endif
