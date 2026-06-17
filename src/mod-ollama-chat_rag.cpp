#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace
{
constexpr char const* kRagFallbackPaths[] = {
    "/azerothcore/modules/mod-ollama-chat/data/rag/",
    "../../../modules/mod-ollama-chat/data/rag/",
    "modules/mod-ollama-chat/data/rag/",
    "data/rag/",
    "rag/",
};

bool RagDirHasJson(std::string const& path)
{
    try
    {
        if (!fs::exists(path) || !fs::is_directory(path))
            return false;
        for (auto const& e : fs::directory_iterator(path))
            if (e.is_regular_file() && e.path().extension() == ".json")
                return true;
    }
    catch (std::exception const&) {}
    return false;
}

std::string ResolveRagPath(std::string const& configured)
{
    if (!configured.empty() && RagDirHasJson(configured))
        return configured;
    for (char const* fallback : kRagFallbackPaths)
        if (RagDirHasJson(fallback))
            return fallback;
    return configured.empty() ? kRagFallbackPaths[0] : configured;
}

bool TagContains(std::vector<std::string> const& tags, std::string const& needle)
{
    for (auto const& tag : tags)
    {
        if (tag.find(needle) != std::string::npos)
            return true;
    }
    return false;
}
} // namespace

OllamaRAGSystem::OllamaRAGSystem() : m_initialized(false) {}

OllamaRAGSystem::~OllamaRAGSystem() {}

bool OllamaRAGSystem::Initialize()
{
    if (m_initialized)
        return true;

    m_ragEntries.clear();
    m_entryIndex.clear();

    std::string const dataPath = ResolveRagPath(g_RAGDataPath);
    if (dataPath != g_RAGDataPath)
        g_RAGDataPath = dataPath;

    if (!LoadRAGDataFromDirectory(dataPath))
    {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Failed to load RAG data from directory: {}", dataPath);
        return false;
    }

    BuildEntryIndex();

    m_initialized = true;
    LOG_INFO("server.loading", "[Ollama Chat RAG] Initialized with {} entries",
             m_ragEntries.size());

    return true;
}

void OllamaRAGSystem::BuildEntryIndex()
{
    m_entryIndex.clear();
    m_entryIndex.reserve(m_ragEntries.size());

    for (auto const& entry : m_ragEntries)
    {
        EntryIndex idx;
        std::string entryText = entry.title + " " + entry.content;
        for (auto const& keyword : entry.keywords)
            entryText += " " + keyword;

        idx.termFreq = BuildTermFreq(entryText);
        for (auto const& [_, freq] : idx.termFreq)
            idx.normSq += freq * freq;

        idx.factualTagBoost = TagContains(entry.tags, "landmark") || TagContains(entry.tags, "quest")
            || TagContains(entry.tags, "npc") || TagContains(entry.tags, "zone");
        idx.mechanicsTagPenalty = TagContains(entry.tags, "dps") || TagContains(entry.tags, "mechanics")
            || TagContains(entry.tags, "class");

        m_entryIndex.push_back(std::move(idx));
    }
}

bool OllamaRAGSystem::LoadRAGDataFromDirectory(const std::string& directoryPath)
{
    try
    {
        if (!fs::exists(directoryPath))
        {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Directory does not exist: {}", directoryPath);
            return false;
        }

        if (!fs::is_directory(directoryPath))
        {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Path is not a directory: {}", directoryPath);
            return false;
        }

        uint32_t loadedFiles = 0;
        for (const auto& entry : fs::directory_iterator(directoryPath))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".json")
            {
                if (LoadRAGDataFromFile(entry.path().string()))
                    loadedFiles++;
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} JSON files from {}", loadedFiles, directoryPath);
        return loadedFiles > 0;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading directory {}: {}", directoryPath, e.what());
        return false;
    }
}

bool OllamaRAGSystem::LoadRAGDataFromFile(const std::string& filePath)
{
    try
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Cannot open file: {}", filePath);
            return false;
        }

        nlohmann::json jsonData;
        file >> jsonData;

        if (!jsonData.is_array())
        {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] JSON file must contain an array of entries: {}", filePath);
            return false;
        }

        uint32_t entriesLoaded = 0;
        for (const auto& item : jsonData)
        {
            try
            {
                RAGEntry entry;
                entry.id = item.value("id", "");
                entry.title = item.value("title", "");
                entry.content = item.value("content", "");

                if (entry.id.empty() || entry.content.empty())
                {
                    LOG_ERROR("server.loading", "[Ollama Chat RAG] Entry missing required 'id' or 'content' field in file: {}", filePath);
                    continue;
                }

                if (item.contains("keywords") && item["keywords"].is_array())
                {
                    for (const auto& keyword : item["keywords"])
                        entry.keywords.push_back(keyword.get<std::string>());
                }

                if (item.contains("tags") && item["tags"].is_array())
                {
                    for (const auto& tag : item["tags"])
                    {
                        std::string t = tag.get<std::string>();
                        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
                        entry.tags.push_back(std::move(t));
                    }
                }

                m_ragEntries.push_back(std::move(entry));
                entriesLoaded++;
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("server.loading", "[Ollama Chat RAG] Error parsing entry in {}: {}", filePath, e.what());
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} entries from {}", entriesLoaded, filePath);
        return entriesLoaded > 0;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading file {}: {}", filePath, e.what());
        return false;
    }
}

void OllamaRAGSystem::Reload()
{
    m_initialized = false;
    Initialize();
}

std::vector<RAGResult> OllamaRAGSystem::RetrieveRelevantInfo(const std::string& query, uint32_t maxResults, float similarityThreshold)
{
    return RetrieveRelevantInfo(query, maxResults, similarityThreshold, false);
}

std::vector<RAGResult> OllamaRAGSystem::RetrieveRelevantInfo(
    const std::string& query, uint32_t maxResults, float similarityThreshold, bool factualQuery)
{
    std::vector<RAGResult> results;

    if (!m_initialized || query.empty() || maxResults == 0)
        return results;

    auto queryTf = BuildTermFreq(query);
    float queryNormSq = 0.0f;
    for (auto const& [_, freq] : queryTf)
        queryNormSq += freq * freq;

    if (queryNormSq == 0.0f)
        return results;

    for (size_t i = 0; i < m_ragEntries.size(); ++i)
    {
        float similarity = CalculateSimilarity(queryTf, queryNormSq, i, factualQuery);
        if (similarity >= similarityThreshold)
            results.push_back({ &m_ragEntries[i], similarity });
    }

    if (results.size() > maxResults)
    {
        std::partial_sort(
            results.begin(), results.begin() + maxResults, results.end(),
            [](RAGResult const& a, RAGResult const& b) { return a.similarity > b.similarity; });
        results.resize(maxResults);
    }
    else if (results.size() > 1)
    {
        std::sort(results.begin(), results.end(),
            [](RAGResult const& a, RAGResult const& b) { return a.similarity > b.similarity; });
    }

    return results;
}

std::string OllamaRAGSystem::GetFormattedRAGInfo(const std::vector<RAGResult>& results)
{
    if (results.empty())
        return "";

    std::ostringstream ss;
    for (size_t i = 0; i < results.size(); ++i)
    {
        if (i > 0)
            ss << "\n";
        ss << "- " << results[i].entry->title << ": " << results[i].entry->content;
    }

    return ss.str();
}

float OllamaRAGSystem::CalculateSimilarity(
    std::unordered_map<std::string, float> const& queryTf,
    float queryNormSq,
    size_t entryIndex,
    bool factualQuery) const
{
    if (entryIndex >= m_entryIndex.size() || queryNormSq == 0.0f)
        return 0.0f;

    EntryIndex const& idx = m_entryIndex[entryIndex];
    if (idx.normSq == 0.0f)
        return 0.0f;

    float dot = 0.0f;
    for (auto const& [term, qFreq] : queryTf)
    {
        auto it = idx.termFreq.find(term);
        if (it != idx.termFreq.end())
            dot += qFreq * it->second;
    }

    float sim = dot / (std::sqrt(queryNormSq) * std::sqrt(idx.normSq));

    if (factualQuery)
    {
        if (idx.factualTagBoost)
            sim += 0.08f;
        if (idx.mechanicsTagPenalty)
            sim -= 0.05f;
    }

    return std::max(0.0f, sim);
}

std::string OllamaRAGSystem::PreprocessText(const std::string& text) const
{
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    result.erase(std::remove_if(result.begin(), result.end(),
        [](char c) { return std::ispunct(c); }), result.end());
    return result;
}

std::vector<std::string> OllamaRAGSystem::TokenizeText(const std::string& text) const
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token)
    {
        if (!token.empty())
            tokens.push_back(std::move(token));
    }
    return tokens;
}

std::unordered_map<std::string, float> OllamaRAGSystem::BuildTermFreq(std::string const& text) const
{
    auto tokens = TokenizeText(PreprocessText(text));
    std::unordered_map<std::string, float> tf;
    tf.reserve(tokens.size());
    for (auto const& token : tokens)
        tf[token] += 1.0f;
    return tf;
}
