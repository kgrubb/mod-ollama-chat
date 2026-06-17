#include "mod-ollama-chat_verify.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace
{
constexpr uint32_t kReplyVerificationMaxRetries = 1;

std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string FirstNWordsLower(std::string const& text, size_t n)
{
    std::istringstream iss(text);
    std::string word;
    std::string out;
    while (n-- > 0 && iss >> word)
    {
        if (!out.empty())
            out += ' ';
        out += ToLower(word);
    }
    return out;
}

bool ContainsAny(std::string const& haystack, char const* const* markers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (haystack.find(markers[i]) != std::string::npos)
            return true;
    }
    return false;
}

bool HasIntent(ChatIntent const& intent)
{
    return intent.presenceQuestion || intent.partyInvite || intent.botReference;
}
} // namespace

VerifyResult VerifyReply(ChatIntent const& intent, std::string const& triggerMsg,
    std::string const& reply, ChatChannelSourceLocal /*channel*/)
{
    VerifyResult result;
    if (!HasIntent(intent) || reply.empty())
        return result;

    std::string const replyLower = ToLower(reply);
    std::string const head = FirstNWordsLower(reply, 8);
    std::string const triggerLower = ToLower(triggerMsg);

    if (intent.partyInvite)
    {
        static char const* kAcceptDecline[] = {
            "sure", "ok", "yes", "yeah", "yep", "nah", "nope", "cant", "can't",
            "busy", "pass", "decline", "invite me", "inv me"
        };
        if (!ContainsAny(head, kAcceptDecline, sizeof(kAcceptDecline) / sizeof(kAcceptDecline[0])))
        {
            result.pass = false;
            result.feedback = "Accept or decline the party invite in the first few words.";
            return result;
        }
    }

    if (intent.presenceQuestion)
    {
        static char const* kGuide[] = { "quest", "windfury", "lfm", "mobs by", "coords", " xp" };
        bool triggerHasGuide = ContainsAny(triggerLower, kGuide, sizeof(kGuide) / sizeof(kGuide[0]));
        if (!triggerHasGuide &&
            ContainsAny(replyLower, kGuide, sizeof(kGuide) / sizeof(kGuide[0])))
        {
            result.pass = false;
            result.feedback = "Answer presence only. Do not mention quests unless asked.";
            return result;
        }
    }

    if (intent.botReference && !intent.referencedName.empty())
    {
        if (replyLower.find(intent.referencedName) == std::string::npos)
        {
            result.pass = false;
            result.feedback = "Acknowledge " + intent.referencedName + " and what the player said about them.";
            return result;
        }
    }

    return result;
}
