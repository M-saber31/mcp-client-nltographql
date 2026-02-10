using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Samples.Azure.Database.NL2SQL.Models;

public class ChatSession
{
    public required string SessionId { get; init; }
    public required ChatHistory ChatHistory { get; init; }
    public required Kernel Kernel { get; init; }
    public required IChatCompletionService ChatCompletion { get; init; }
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    public DateTime LastActivityAt { get; set; } = DateTime.UtcNow;
}
