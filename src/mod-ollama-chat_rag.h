#ifndef MOD_OLLAMA_CHAT_RAG_H
#define MOD_OLLAMA_CHAT_RAG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>

struct RAGEntry {
    std::string id;
    std::string title;
    std::string content;
    std::vector<std::string> keywords;
    std::vector<std::string> tags;
    std::vector<std::string> acronyms;
    uint32_t botLevelMin = 0;
    uint32_t botLevelMax = 999;
};

struct RAGResult {
    const RAGEntry* entry;
    float similarity;
};

struct DungeonMatch
{
    std::string id;
    std::string fullName;
    uint32_t levelMin = 0;
    uint32_t levelMax = 0;
    RAGEntry const* entry = nullptr;
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
    RAGEntry const* GetEntryById(std::string const& id) const;

    std::vector<DungeonMatch> ResolveAcronyms(std::string const& message, uint32_t botLevel) const;
    std::string FormatEligibilityHint(std::vector<DungeonMatch> const& matches, uint32_t botLevel) const;

private:
    struct EntryIndex {
        std::unordered_map<std::string, float> termFreq;
        float normSq = 0.0f;
        bool factualTagBoost = false;
        bool mechanicsTagPenalty = false;
    };

    struct AcronymEntry
    {
        std::string id;
        std::string fullName;
        uint32_t levelMin = 0;
        uint32_t levelMax = 0;
        uint32_t botLevelMin = 0;
        uint32_t botLevelMax = 999;
        RAGEntry const* source = nullptr;
        std::vector<std::string> acronyms;
    };

    struct PhraseAcronym
    {
        std::string phrase;
        size_t entryIndex = 0;
    };

    bool LoadRAGDataFromDirectory(const std::string& directoryPath);
    bool LoadRAGDataFromFile(const std::string& filePath);
    void BuildEntryIndex();
    void BuildAcronymIndex();
    size_t PickBestAcronymEntry(std::vector<size_t> const& indices, uint32_t botLevel) const;

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
    std::unordered_map<std::string, size_t> m_entryIdIndex;
    std::unordered_map<std::string, std::vector<size_t>> m_termToEntries;
    std::vector<AcronymEntry> m_acronymEntries;
    std::unordered_map<std::string, std::vector<size_t>> m_wordAcronyms;
    std::vector<PhraseAcronym> m_phraseAcronyms;
    std::unordered_set<std::string> m_ambiguousAcronyms;
    bool m_initialized;
};

#endif // MOD_OLLAMA_CHAT_RAG_H
