#include "mod-ollama-chat_querymanager.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include <thread>
#include <utility>

QueryManager::QueryManager()
    : maxConcurrentQueries(g_MaxConcurrentQueries), currentQueries(0)
{
}

void QueryManager::setMaxConcurrentQueries(int maxQueries) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxConcurrentQueries = maxQueries;
}

std::future<std::string> QueryManager::submitQuery(std::string const& prompt)
{
    PromptBundle bundle;
    bundle.user = prompt;
    return submitQuery(std::move(bundle));
}

std::future<std::string> QueryManager::submitQuery(PromptBundle bundle)
{
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries) {
            ++currentQueries;
        } else {
            taskQueue.push({ std::move(bundle), std::move(promise) });
            return future;
        }
    }

    std::thread(&QueryManager::processQuery, this, std::move(bundle), std::move(promise)).detach();
    return future;
}

void QueryManager::processQuery(PromptBundle bundle, std::promise<std::string> promise)
{
    std::string result;
    try
    {
        result = QueryOllamaAPI(bundle);
    }
    catch (...)
    {
        result = "";
    }

    try
    {
        promise.set_value(result);
    }
    catch (...)
    {
    }

    std::lock_guard<std::mutex> lock(mutex_);
    --currentQueries;
    if (!taskQueue.empty() && (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries))
    {
        QueryTask task = std::move(taskQueue.front());
        taskQueue.pop();
        ++currentQueries;
        std::thread(&QueryManager::processQuery, this, std::move(task.bundle), std::move(task.promise)).detach();
    }
}
