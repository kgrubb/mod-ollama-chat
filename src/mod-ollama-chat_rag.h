#ifndef MOD_OLLAMA_CHAT_RAG_H
#define MOD_OLLAMA_CHAT_RAG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

struct RAGEntry {
    std::string id;
    std::string title;
    std::string content;
    std::vector<std::string> keywords;
    std::vector<std::string> tags;
};

struct RAGResult {
    const RAGEntry* entry;
    float similarity;
};

class OllamaRAGSystem {
public:
    OllamaRAGSystem();
    ~OllamaRAGSystem();

    bool Initialize();

    std::vector<RAGResult> RetrieveRelevantInfo(const std::string& query, uint32_t maxResults = 3, float similarityThreshold = 0.3f);
    std::vector<RAGResult> RetrieveRelevantInfo(const std::string& query, uint32_t maxResults, float similarityThreshold, bool factualQuery);

    void Reload();

    std::string GetFormattedRAGInfo(const std::vector<RAGResult>& results);

private:
    struct EntryIndex {
        std::unordered_map<std::string, float> termFreq;
        float normSq = 0.0f;
        bool factualTagBoost = false;
        bool mechanicsTagPenalty = false;
    };

    bool LoadRAGDataFromDirectory(const std::string& directoryPath);
    bool LoadRAGDataFromFile(const std::string& filePath);
    void BuildEntryIndex();

    float CalculateSimilarity(
        std::unordered_map<std::string, float> const& queryTf,
        float queryNormSq,
        size_t entryIndex,
        bool factualQuery) const;

    std::string PreprocessText(const std::string& text) const;
    std::vector<std::string> TokenizeText(const std::string& text) const;
    std::unordered_map<std::string, float> BuildTermFreq(std::string const& text) const;

private:
    std::vector<RAGEntry> m_ragEntries;
    std::vector<EntryIndex> m_entryIndex;
    bool m_initialized;
};

#endif // MOD_OLLAMA_CHAT_RAG_H
