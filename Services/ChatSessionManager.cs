using System.Collections.Concurrent;
using Microsoft.SemanticKernel.ChatCompletion;
using Samples.Azure.Database.NL2SQL.Models;

namespace Samples.Azure.Database.NL2SQL.Services;

public class ChatSessionManager
{
    private readonly ConcurrentDictionary<string, ChatSession> _sessions = new();
    private readonly IKernelFactory _kernelFactory;
    private string _schemaSummary = string.Empty;

    public ChatSessionManager(IKernelFactory kernelFactory)
    {
        _kernelFactory = kernelFactory;
    }

    public void SetSchemaSummary(string summary) => _schemaSummary = summary;

    public ChatSession CreateSession()
    {
        var sessionId = Guid.NewGuid().ToString("N");
        var (kernel, ai) = _kernelFactory.CreateKernelForSession();

        var systemPrompt = $"""
            You are an AI assistant that helps users query data via a GraphQL API.
            The API exposes the following queries:

            {_schemaSummary}

            Use a professional tone when answering and provide a summary of data instead of lists.
            If users ask about topics you don't know, answer that you don't know. Today's date is {DateTime.Now:yyyy-MM-dd}.
            You must use the provided QueryGraphQL tool to query the GraphQL API with valid GraphQL query strings.
            If you need more details about the schema structure or available fields, use the IntrospectSchema tool.
            If the request is complex, break it down into smaller steps and call the QueryGraphQL tool as many times as needed.
            """;

        var session = new ChatSession
        {
            SessionId = sessionId,
            ChatHistory = new ChatHistory(systemPrompt),
            Kernel = kernel,
            ChatCompletion = ai
        };

        _sessions.TryAdd(sessionId, session);
        return session;
    }

    public ChatSession? GetSession(string sessionId)
    {
        _sessions.TryGetValue(sessionId, out var session);
        return session;
    }

    public bool DeleteSession(string sessionId)
    {
        return _sessions.TryRemove(sessionId, out _);
    }

    public IEnumerable<ChatSession> ListSessions()
    {
        return _sessions.Values.OrderByDescending(s => s.LastActivityAt);
    }

    public bool ClearHistory(string sessionId)
    {
        if (!_sessions.TryGetValue(sessionId, out var session))
            return false;

        session.ChatHistory.RemoveRange(1, session.ChatHistory.Count - 1);
        return true;
    }
}
