using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.AspNetCore.SignalR;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Samples.Azure.Database.NL2SQL.Services;

#pragma warning disable SKEXP0010

namespace Samples.Azure.Database.NL2SQL.Hubs;

public class ChatHub : Hub
{
    private readonly ChatSessionManager _sessionManager;
    private readonly ILogger<ChatHub> _logger;

    public ChatHub(ChatSessionManager sessionManager, ILogger<ChatHub> logger)
    {
        _sessionManager = sessionManager;
        _logger = logger;
    }

    public async IAsyncEnumerable<string> SendMessage(
        string sessionId,
        string message,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var session = _sessionManager.GetSession(sessionId);
        if (session == null)
            throw new HubException("Session not found. Create a session first via the REST API.");

        session.LastActivityAt = DateTime.UtcNow;
        session.ChatHistory.AddUserMessage(message);

        var settings = new AzureOpenAIPromptExecutionSettings
        {
            FunctionChoiceBehavior = Microsoft.SemanticKernel.FunctionChoiceBehavior.Auto()
        };

        var fullResponse = new StringBuilder();

        _logger.LogInformation("Session {SessionId}: Processing message", sessionId);

        await foreach (var chunk in session.ChatCompletion
            .GetStreamingChatMessageContentsAsync(
                session.ChatHistory, settings, session.Kernel, cancellationToken))
        {
            if (chunk.Content != null)
            {
                fullResponse.Append(chunk.Content);
                yield return chunk.Content;
            }
        }

        session.ChatHistory.AddAssistantMessage(fullResponse.ToString());
        session.LastActivityAt = DateTime.UtcNow;

        _logger.LogInformation("Session {SessionId}: Response complete", sessionId);
    }
}
