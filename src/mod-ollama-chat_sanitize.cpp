#include "mod-ollama-chat_sanitize.h"
#include <algorithm>
#include <cctype>

namespace
{
void TrimInPlace(std::string& s)
{
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
}

void LowerInPlace(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

size_t CountWords(std::string const& s)
{
    size_t n = 0;
    bool inWord = false;
    for (unsigned char c : s)
    {
        if (std::isspace(c))
            inWord = false;
        else if (!inWord)
        {
            inWord = true;
            ++n;
        }
    }
    return n;
}
} // namespace

bool FinalizeGeneralChatLine(std::string& line, std::string const& triggerMsg)
{
    TrimInPlace(line);
    if (line.empty())
        return false;

    if (line.size() >= 2 && line.front() == '"' && line.back() == '"')
    {
        line = line.substr(1, line.size() - 2);
        TrimInPlace(line);
        if (line.empty())
            return false;
    }

    for (char& c : line)
    {
        if (c == '\n' || c == '\r')
            c = ' ';
    }

    if (CountWords(line) > 15)
        return false;

    if (line.find('*') != std::string::npos)
        return false;

    if (!triggerMsg.empty())
    {
        std::string lower = line;
        LowerInPlace(lower);
        std::string triggerLower = triggerMsg;
        LowerInPlace(triggerLower);
        TrimInPlace(triggerLower);
        if (lower == triggerLower)
            return false;
    }

    return true;
}
