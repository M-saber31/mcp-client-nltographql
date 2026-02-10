using Samples.Azure.Database.NL2SQL.Services;

namespace Samples.Azure.Database.NL2SQL.Endpoints;

public static class ChatEndpoints
{
    public static void MapChatEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/chat");

        group.MapPost("/sessions", (ChatSessionManager mgr) =>
        {
            var session = mgr.CreateSession();
            return Results.Ok(new { sessionId = session.SessionId, createdAt = session.CreatedAt });
        });

        group.MapGet("/sessions/{id}", (string id, ChatSessionManager mgr) =>
        {
            var session = mgr.GetSession(id);
            return session == null
                ? Results.NotFound()
                : Results.Ok(new
                {
                    sessionId = session.SessionId,
                    createdAt = session.CreatedAt,
                    lastActivityAt = session.LastActivityAt,
                    messageCount = session.ChatHistory.Count - 1
                });
        });

        group.MapGet("/sessions/{id}/history", (string id, ChatSessionManager mgr) =>
        {
            var session = mgr.GetSession(id);
            if (session == null) return Results.NotFound();

            var messages = session.ChatHistory
                .Skip(1)
                .Select(m => new { role = m.Role.ToString().ToLower(), content = m.Content });
            return Results.Ok(messages);
        });

        group.MapDelete("/sessions/{id}", (string id, ChatSessionManager mgr) =>
        {
            return mgr.DeleteSession(id) ? Results.NoContent() : Results.NotFound();
        });

        group.MapDelete("/sessions/{id}/history", (string id, ChatSessionManager mgr) =>
        {
            return mgr.ClearHistory(id) ? Results.NoContent() : Results.NotFound();
        });

        app.MapGet("/api/health", () => Results.Ok(new { status = "ok" }));
    }
}
