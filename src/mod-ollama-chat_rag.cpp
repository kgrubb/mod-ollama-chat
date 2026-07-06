#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <climits>
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

std::string RagToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool RagContainsWholeWord(std::string const& haystackLower, std::string const& wordLower)
{
    if (wordLower.empty())
        return false;
    size_t pos = 0;
    while ((pos = haystackLower.find(wordLower, pos)) != std::string::npos)
    {
        bool leftOk = pos == 0 || !std::isalnum(static_cast<unsigned char>(haystackLower[pos - 1]));
        size_t end = pos + wordLower.size();
        bool rightOk = end >= haystackLower.size() ||
            !std::isalnum(static_cast<unsigned char>(haystackLower[end]));
        if (leftOk && rightOk)
            return true;
        pos += wordLower.size();
    }
    return false;
}

bool ParseLevelTag(std::vector<std::string> const& tags, uint32_t& minOut, uint32_t& maxOut)
{
    for (std::string const& tag : tags)
    {
        if (tag.rfind("level:", 0) != 0)
            continue;
        size_t dash = tag.find('-', 6);
        if (dash == std::string::npos)
            continue;
        char* end = nullptr;
        long minLevel = std::strtol(tag.c_str() + 6, &end, 10);
        if (end == tag.c_str() + 6)
            continue;
        long maxLevel = std::strtol(tag.c_str() + dash + 1, &end, 10);
        if (end == tag.c_str() + dash + 1)
            continue;
        minOut = static_cast<uint32_t>(minLevel);
        maxOut = static_cast<uint32_t>(maxLevel);
        return true;
    }
    return false;
}

void CollectMessageTokens(std::string const& lower, std::unordered_set<std::string>& out)
{
    std::string token;
    for (unsigned char c : lower)
    {
        if (std::isalnum(c))
            token += static_cast<char>(c);
        else if (!token.empty())
        {
            out.insert(std::move(token));
            token.clear();
        }
    }
    if (!token.empty())
        out.insert(std::move(token));
}

bool LooksDungeonRelated(std::string const& lower)
{
    static char const* markers[] = {
        " lfg", "lfm", "lf1", "lf2", "lf3", "looking for", "group", "invite",
        "run", "running", "dungeon", "instance", "heroic", "normal", "level", "lvl"
    };
    std::string const padded = " " + lower;
    for (char const* marker : markers)
        if (padded.find(marker) != std::string::npos)
            return true;
    return false;
}

bool AcronymNeedsContext(std::string const& acronym)
{
    return acronym.size() <= 2 || acronym == "st" || acronym == "stock";
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
    m_entryIdIndex.clear();
    m_termToEntries.clear();
    m_acronymEntries.clear();
    m_wordAcronyms.clear();
    m_phraseAcronyms.clear();
    m_ambiguousAcronyms.clear();

    std::string const dataPath = ResolveRagPath(g_RAGDataPath);
    if (dataPath != g_RAGDataPath)
        g_RAGDataPath = dataPath;

    if (!LoadRAGDataFromDirectory(dataPath))
    {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Failed to load RAG data from directory: {}", dataPath);
        BuildAcronymIndex();
        return false;
    }

    BuildEntryIndex();
    BuildAcronymIndex();

    m_initialized = true;
    LOG_INFO("server.loading", "[Ollama Chat RAG] Initialized with {} entries",
             m_ragEntries.size());

    return true;
}

void OllamaRAGSystem::BuildEntryIndex()
{
    m_entryIndex.clear();
    m_entryIdIndex.clear();
    m_termToEntries.clear();
    m_entryIndex.reserve(m_ragEntries.size());

    for (size_t i = 0; i < m_ragEntries.size(); ++i)
    {
        RAGEntry const& entry = m_ragEntries[i];
        EntryIndex idx;
        std::string entryText = entry.title + " " + entry.content;
        for (auto const& keyword : entry.keywords)
            entryText += " " + keyword;

        idx.termFreq = BuildTermFreq(entryText);
        for (auto const& [term, freq] : idx.termFreq)
        {
            idx.normSq += freq * freq;
            m_termToEntries[term].push_back(i);
        }

        idx.factualTagBoost = TagContains(entry.tags, "landmark") || TagContains(entry.tags, "quest")
            || TagContains(entry.tags, "npc") || TagContains(entry.tags, "zone");
        idx.mechanicsTagPenalty = TagContains(entry.tags, "dps") || TagContains(entry.tags, "mechanics")
            || TagContains(entry.tags, "class");

        m_entryIndex.push_back(std::move(idx));
        m_entryIdIndex.emplace(entry.id, i);
    }
}

void OllamaRAGSystem::BuildAcronymIndex()
{
    m_acronymEntries.clear();
    m_wordAcronyms.clear();
    m_phraseAcronyms.clear();
    m_ambiguousAcronyms.clear();
    m_acronymEntries.reserve(80);

    for (size_t i = 0; i < m_ragEntries.size(); ++i)
    {
        RAGEntry const& e = m_ragEntries[i];
        if (e.acronyms.empty())
            continue;

        AcronymEntry row;
        row.id = e.id;
        row.fullName = e.title;
        row.botLevelMin = e.botLevelMin;
        row.botLevelMax = e.botLevelMax;
        row.source = &m_ragEntries[i];
        row.acronyms.reserve(e.acronyms.size());
        for (std::string const& ac : e.acronyms)
        {
            std::string lower = RagToLower(ac);
            if (!lower.empty())
                row.acronyms.push_back(std::move(lower));
        }
        if (row.acronyms.empty())
            continue;

        if (!ParseLevelTag(e.tags, row.levelMin, row.levelMax))
            continue;

        size_t const idx = m_acronymEntries.size();
        m_acronymEntries.push_back(std::move(row));

        for (std::string const& ac : m_acronymEntries[idx].acronyms)
        {
            if (ac.find(' ') != std::string::npos)
                m_phraseAcronyms.push_back({ ac, idx });
            else
                m_wordAcronyms[ac].push_back(idx);
        }
    }

    for (auto const& [ac, indices] : m_wordAcronyms)
    {
        if (indices.size() > 1)
            m_ambiguousAcronyms.insert(ac);
    }
}

size_t OllamaRAGSystem::PickBestAcronymEntry(std::vector<size_t> const& indices, uint32_t botLevel) const
{
    size_t bestIdx = indices.front();
    int bestScore = INT_MAX;

    for (size_t idx : indices)
    {
        AcronymEntry const& e = m_acronymEntries[idx];
        int score = 0;
        if (botLevel >= e.levelMin && botLevel <= e.levelMax)
            score = 0;
        else if (botLevel < e.levelMin)
            score = static_cast<int>(e.levelMin - botLevel);
        else
            score = static_cast<int>(botLevel - e.levelMax);

        if (botLevel < e.botLevelMin || (e.botLevelMax < 999 && botLevel > e.botLevelMax))
            score += 1000;

        if (score < bestScore)
        {
            bestScore = score;
            bestIdx = idx;
        }
    }

    return bestIdx;
}

std::vector<DungeonMatch> OllamaRAGSystem::ResolveAcronyms(std::string const& message, uint32_t botLevel) const
{
    std::vector<DungeonMatch> out;
    if (message.empty() || m_acronymEntries.empty())
        return out;

    std::string const lower = RagToLower(message);
    bool const dungeonRelated = LooksDungeonRelated(lower);
    std::unordered_map<std::string, std::vector<size_t>> hits;
    hits.reserve(4);

    auto addHit = [&](std::string const& token, size_t entryIdx) {
        auto& vec = hits[token];
        if (std::find(vec.begin(), vec.end(), entryIdx) == vec.end())
            vec.push_back(entryIdx);
    };

    std::unordered_set<std::string> tokens;
    tokens.reserve(8);
    CollectMessageTokens(lower, tokens);
    for (std::string const& token : tokens)
    {
        auto it = m_wordAcronyms.find(token);
        if (it == m_wordAcronyms.end())
            continue;
        for (size_t idx : it->second)
        {
            AcronymEntry const& e = m_acronymEntries[idx];
            for (std::string const& ac : e.acronyms)
            {
                if (ac != token)
                    continue;
                if (AcronymNeedsContext(ac) && !dungeonRelated)
                    continue;
                if (RagContainsWholeWord(lower, ac))
                    addHit(token, idx);
                break;
            }
        }
    }

    for (PhraseAcronym const& phrase : m_phraseAcronyms)
    {
        if (AcronymNeedsContext(phrase.phrase) && !dungeonRelated)
            continue;
        if (RagContainsWholeWord(lower, phrase.phrase))
            addHit(phrase.phrase, phrase.entryIndex);
    }

    if (hits.empty())
        return out;

    out.reserve(hits.size());
    std::unordered_set<std::string> seenIds;
    seenIds.reserve(hits.size());

    for (auto& [token, indices] : hits)
    {
        if (indices.empty())
            continue;

        if (indices.size() > 1 || m_ambiguousAcronyms.count(token))
            indices = { PickBestAcronymEntry(indices, botLevel) };

        for (size_t idx : indices)
        {
            AcronymEntry const& e = m_acronymEntries[idx];
            if (seenIds.count(e.id))
                continue;
            seenIds.insert(e.id);
            out.push_back({ e.id, e.fullName, e.levelMin, e.levelMax, e.source });
        }
    }

    return out;
}

std::string OllamaRAGSystem::FormatEligibilityHint(std::vector<DungeonMatch> const& matches, uint32_t botLevel) const
{
    if (matches.empty())
        return "";

    std::ostringstream hint;
    hint << "Use the listed level range as fact. Do not invent other ranges.\n"
         << "Only claim ineligibility if your level is outside that range.\n"
         << "If uninterested, decline honestly. Never fake a level excuse.\n";
    for (size_t i = 0; i < matches.size(); ++i)
    {
        DungeonMatch const& m = matches[i];
        bool eligible = botLevel >= m.levelMin && botLevel <= m.levelMax;
        hint << m.fullName << " is levels " << m.levelMin << "-" << m.levelMax
             << ". You are L" << botLevel << (eligible ? ". Eligible." : ". Outside range.");
        if (i + 1 < matches.size())
            hint << "\n";
    }
    return hint.str();
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

                if (item.contains("acronyms") && item["acronyms"].is_array())
                {
                    for (const auto& acronym : item["acronyms"])
                        entry.acronyms.push_back(acronym.get<std::string>());
                }

                entry.botLevelMin = item.value("bot_level_min", 0u);
                entry.botLevelMax = item.value("bot_level_max", 999u);

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

RAGEntry const* OllamaRAGSystem::GetEntryById(std::string const& id) const
{
    auto it = m_entryIdIndex.find(id);
    if (it == m_entryIdIndex.end())
        return nullptr;
    return &m_ragEntries[it->second];
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

    std::unordered_set<size_t> candidates;
    candidates.reserve(queryTf.size() * 4);
    for (auto const& [term, _] : queryTf)
    {
        auto it = m_termToEntries.find(term);
        if (it == m_termToEntries.end())
            continue;
        for (size_t idx : it->second)
            candidates.insert(idx);
    }

    if (candidates.empty())
        return results;

    for (size_t i : candidates)
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
