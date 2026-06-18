#include "mod-ollama-chat_verify.h"
#include <algorithm>
#include <cctype>

namespace
{
std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool HasIntent(ChatIntent const& intent)
{
    return intent.presenceQuestion || intent.partyInvite || intent.botReference;
}
} // namespace

VerifyResult VerifyReply(ChatIntent const& intent, std::string const& /*triggerMsg*/,
    std::string const& reply, ChatChannelSourceLocal /*channel*/)
{
    VerifyResult result;
    if (!HasIntent(intent) || reply.empty())
        return result;

    if (intent.botReference && !intent.referencedName.empty())
    {
        std::string const replyLower = ToLower(reply);
        if (replyLower.find(intent.referencedName) == std::string::npos)
        {
            result.pass = false;
            result.feedback = "Acknowledge " + intent.referencedName + " and what the player said about them.";
        }
    }

    return result;
}
