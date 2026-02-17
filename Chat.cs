using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Agents.AI;
using Azure.AI.OpenAI;
using Azure.Identity;
using OpenAI.Chat;
using ModelContextProtocol.Client;
using DotNetEnv;
using Spectre.Console;

namespace Samples.Azure.Database.NL2SQL;

/// <summary>
/// Wraps an AIFunction to measure execution time and truncate large results.
/// Truncation happens BEFORE the result is sent back to the LLM, reducing
/// the token count the LLM must process on the current turn.
/// </summary>
public class TimedAIFunction : AIFunction
{
    private readonly AIFunction _inner;

    public TimedAIFunction(AIFunction inner) => _inner = inner;

    public override string Name => _inner.Name;
    public override string Description => _inner.Description;
    public override JsonElement JsonSchema => _inner.JsonSchema;
    public override JsonSerializerOptions JsonSerializerOptions => _inner.JsonSerializerOptions;

    protected override async ValueTask<object?> InvokeCoreAsync(
        AIFunctionArguments arguments,
        CancellationToken cancellationToken)
    {
        var sw = Stopwatch.StartNew();
        var result = await _inner.InvokeAsync(arguments, cancellationToken);
        sw.Stop();
        AnsiConsole.MarkupLine($"  [yellow]>> Tool '{Markup.Escape(Name)}' completed in {sw.Elapsed.TotalSeconds:F2}s[/]");
        return result;
    }
}

/// <summary>
/// IChatClient middleware that measures and logs the duration of each LLM API call.
/// </summary>
public class TimingChatClient : DelegatingChatClient
{
    private int _callIndex = 0;

    public TimingChatClient(IChatClient inner) : base(inner) { }

    public override async Task<Microsoft.Extensions.AI.ChatResponse> GetResponseAsync(
        IEnumerable<Microsoft.Extensions.AI.ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var callNum = Interlocked.Increment(ref _callIndex);
        var messageCount = messages.Count();
        var sw = Stopwatch.StartNew();
        AnsiConsole.MarkupLine($"  [blue]>> LLM call #{callNum} started ({messageCount} messages in history)...[/]");

        var response = await base.GetResponseAsync(messages, options, cancellationToken);
        sw.Stop();

        var hasToolCalls = response.Messages
            .SelectMany(m => m.Contents)
            .Any(c => c is FunctionCallContent);
        var label = hasToolCalls ? "tool call request" : "text response";

        AnsiConsole.MarkupLine($"  [blue]>> LLM call #{callNum} completed in {sw.Elapsed.TotalSeconds:F2}s ({label})[/]");
        return response;
    }

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<Microsoft.Extensions.AI.ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var callNum = Interlocked.Increment(ref _callIndex);
        var messageCount = messages.Count();
        var sw = Stopwatch.StartNew();
        var firstToken = true;
        AnsiConsole.MarkupLine($"  [blue]>> LLM call #{callNum} started ({messageCount} messages in history)...[/]");

        await foreach (var update in base.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            if (firstToken && sw.IsRunning)
            {
                AnsiConsole.MarkupLine($"  [blue]>> LLM call #{callNum} first token in {sw.Elapsed.TotalSeconds:F2}s[/]");
                firstToken = false;
            }
            yield return update;
        }
        sw.Stop();
        AnsiConsole.MarkupLine($"  [blue]>> LLM call #{callNum} stream completed in {sw.Elapsed.TotalSeconds:F2}s[/]");
    }
}

public class ChatBot
{
    const string VERSION = "3.0";
    private readonly string azureOpenAIEndpoint;
    private readonly string azureOpenAIApiKey;
    private readonly string chatModelDeploymentName;
    private readonly string mcpServerUrl;

    public ChatBot(string envFile)
    {
        Env.Load(envFile);
        azureOpenAIEndpoint = Env.GetString("OPENAI_URL");
        azureOpenAIApiKey = Env.GetString("OPENAI_KEY") ?? string.Empty;
        chatModelDeploymentName = Env.GetString("OPENAI_CHAT_DEPLOYMENT_NAME");
        mcpServerUrl = Env.GetString("MCP_SERVER_URL");
    }

    public async Task RunAsync(bool enableDebug = false)
    {
        AnsiConsole.Clear();
        AnsiConsole.Foreground = Color.Green;

        var table = new Table();
        table.Expand();
        table.AddColumn(new TableColumn($"[bold]Natural Language GraphQL Chatbot Agent[/] v{VERSION}").Centered());
        AnsiConsole.Write(table);

        var loggerFactory = LoggerFactory.Create(b =>
        {
            b.ClearProviders();
            b.SetMinimumLevel(enableDebug ? LogLevel.Trace : LogLevel.None);
            b.AddProvider(new SpectreConsoleLoggerProvider());
        });

        (var agent, var mcpClient, var session) = await AnsiConsole.Status().StartAsync("Booting up agents...", async ctx =>
        {
            ctx.Spinner(Spinner.Known.Default);
            ctx.SpinnerStyle(Style.Parse("yellow"));

            var sw = Stopwatch.StartNew();

            // Connect to MCP server
            AnsiConsole.WriteLine("Connecting to MCP server...");
            AnsiConsole.WriteLine($"MCP Server URL: {mcpServerUrl}");

            var transport = new SseClientTransport(new SseClientTransportOptions
            {
                Endpoint = new Uri(mcpServerUrl),
                Name = "nl2graphql-mcp-server",
                TransportMode = HttpTransportMode.AutoDetect
            }, loggerFactory: loggerFactory);

            var mcpClient = await McpClientFactory.CreateAsync(transport, loggerFactory: loggerFactory);
            AnsiConsole.MarkupLine($"[yellow]MCP session established. ({sw.Elapsed.TotalSeconds:F2}s)[/]");
            sw.Restart();

            // Discover MCP tools
            AnsiConsole.WriteLine("Discovering MCP tools...");
            var mcpTools = await mcpClient.ListToolsAsync();
            foreach (var tool in mcpTools)
            {
                AnsiConsole.WriteLine($"  Tool: {tool.Name} - {tool.Description}");
            }
            AnsiConsole.MarkupLine($"[yellow]Tool discovery completed. ({sw.Elapsed.TotalSeconds:F2}s)[/]");
            sw.Restart();

            // Introspect schema for the system prompt
            AnsiConsole.WriteLine("Introspecting GraphQL schema...");
            var introspectTool = mcpTools.FirstOrDefault(t => t.Name == "introspect-schema");
            string schemaSummary;
            if (introspectTool != null)
            {
                var schemaResult = await mcpClient.CallToolAsync("introspect-schema");
                var schemaJson = schemaResult.Content
                    .Where(c => c.Type == "text")
                    .Select(c => c.Text)
                    .FirstOrDefault() ?? string.Empty;
                schemaSummary = ExtractSchemaSummary(schemaJson);
            }
            else
            {
                schemaSummary = "Schema not available - use introspection tools if needed.";
            }
            AnsiConsole.MarkupLine($"[yellow]Schema introspection completed. ({sw.Elapsed.TotalSeconds:F2}s)[/]");
            sw.Restart();

            // Create the Azure OpenAI agent with MCP tools and schema baked into instructions
            AnsiConsole.WriteLine("Initializing agent...");

            var azureClient = string.IsNullOrEmpty(azureOpenAIApiKey)
                ? new AzureOpenAIClient(new Uri(azureOpenAIEndpoint), new DefaultAzureCredential())
                : new AzureOpenAIClient(new Uri(azureOpenAIEndpoint), new System.ClientModel.ApiKeyCredential(azureOpenAIApiKey));

            var chatClient = azureClient.GetChatClient(chatModelDeploymentName);

            var agent = chatClient.AsAIAgent(
                instructions: $"""
                    You are an AI assistant that helps users query data via a GraphQL API.
                    The API exposes the following queries:

                    {schemaSummary}

                    Use a professional tone when answering and provide a summary of data instead of lists.
                    If users ask about topics you don't know, answer that you don't know. Today's date is {DateTime.Now:yyyy-MM-dd}.
                    You must use the provided query-graphql tool to query the GraphQL API with valid GraphQL query strings.
                    The schema above is complete — do NOT call introspect-schema, it is not available as a tool.
                    IMPORTANT: Always use pagination arguments (e.g. first: 20) to limit query results. Never fetch all rows at once.
                    Only request the specific fields needed to answer the user's question — do not select all fields.
                    If the user needs aggregate data (counts, totals), prefer using available aggregate queries over fetching raw rows.
                    """,
                name: "GraphQLAssistant",
                tools: mcpTools
                    .Where(t => t.Name != "introspect-schema")
                    .Select(t => (AITool)new TimedAIFunction(t))
                    .ToList(),
                clientFactory: inner => new TimingChatClient(inner),
                loggerFactory: loggerFactory
            );

            AnsiConsole.MarkupLine($"[yellow]Agent initialized. ({sw.Elapsed.TotalSeconds:F2}s)[/]");

            // Create a session to manage conversation state
            var session = await agent.CreateSessionAsync();

            AnsiConsole.WriteLine("All done.");
            return (agent, mcpClient, session);
        });

        AnsiConsole.WriteLine("Ready to chat! Hit 'ctrl-c' to quit.");

        try
        {
            while (true)
            {
                AnsiConsole.WriteLine();
                var question = AnsiConsole.Prompt(new TextPrompt<string>($"🧑: "));

                if (string.IsNullOrWhiteSpace(question))
                    continue;

                switch (question)
                {
                    case "/c":
                        AnsiConsole.Clear();
                        continue;
                    case "/ch":
                        session = await agent.CreateSessionAsync();
                        AnsiConsole.WriteLine("Chat history cleared (new session created).");
                        continue;
                    case "/h":
                        DisplaySessionContents(agent, session);
                        continue;
                }

                AnsiConsole.WriteLine();

                var turnSw = Stopwatch.StartNew();
                AnsiConsole.Write("🤖: ");

                await foreach (var update in agent.RunStreamingAsync(question, session))
                {
                    var text = update.Text;
                    if (!string.IsNullOrEmpty(text))
                    {
                        AnsiConsole.Write(text);
                    }
                }
                turnSw.Stop();
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine($"[yellow]>> Total turn time: {turnSw.Elapsed.TotalSeconds:F2}s[/]");

                // Remove tool call and tool result messages from session history
                // to keep it lean — only user questions and assistant summaries are retained
                StripToolMessagesFromSession(session);
            }
        }
        finally
        {
            if (mcpClient is IAsyncDisposable asyncDisposable)
                await asyncDisposable.DisposeAsync();
            else if (mcpClient is IDisposable disposable)
                disposable.Dispose();
        }
    }

    /// <summary>
    /// Removes all tool call and tool result messages from the session's chat history.
    /// This keeps only user messages and assistant text responses, reducing token usage
    /// on subsequent LLM calls.
    /// </summary>
    private static void StripToolMessagesFromSession(AgentSession session)
    {
        var provider = session.GetService<InMemoryChatHistoryProvider>();
        if (provider == null) return;

        for (int i = provider.Count - 1; i >= 0; i--)
        {
            var msg = provider[i];

            // Remove messages with role "tool" (tool results)
            if (msg.Role == ChatRole.Tool)
            {
                provider.RemoveAt(i);
                continue;
            }

            // Remove assistant messages that contain tool call requests (no text content)
            if (msg.Role == ChatRole.Assistant && msg.Contents != null)
            {
                bool hasToolCalls = msg.Contents.Any(c => c is FunctionCallContent);
                bool hasTextContent = msg.Contents.Any(c => c is TextContent tc && !string.IsNullOrWhiteSpace(tc.Text));

                // If it's purely tool calls with no text, remove it
                if (hasToolCalls && !hasTextContent)
                {
                    provider.RemoveAt(i);
                }
                // If it has both tool calls and text, keep only the text
                else if (hasToolCalls && hasTextContent)
                {
                    var textOnly = msg.Contents.Where(c => c is TextContent).ToList();
                    provider[i] = new Microsoft.Extensions.AI.ChatMessage(ChatRole.Assistant, textOnly);
                }
            }
        }
    }

    /// <summary>
    /// Displays the full contents of the AgentSession for inspection.
    /// Shows the serialized JSON structure and a human-readable message breakdown.
    /// </summary>
    private static void DisplaySessionContents(ChatClientAgent agent, AgentSession session)
    {
        var provider = session.GetService<InMemoryChatHistoryProvider>();
        if (provider == null)
        {
            AnsiConsole.MarkupLine("[red]Could not access session history provider.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"[bold yellow]--- Session Contents ({provider.Count} messages) ---[/]");
        AnsiConsole.WriteLine();

        for (int i = 0; i < provider.Count; i++)
        {
            var msg = provider[i];
            var roleColor = msg.Role == ChatRole.User ? "green"
                : msg.Role == ChatRole.Assistant ? "cyan"
                : msg.Role == ChatRole.Tool ? "yellow"
                : "white";

            AnsiConsole.MarkupLine($"[bold {roleColor}]Message {i}: Role={msg.Role}[/]");

            // Show content types
            if (msg.Contents != null)
            {
                foreach (var content in msg.Contents)
                {
                    switch (content)
                    {
                        case TextContent tc:
                            var preview = tc.Text?.Length > 200 ? tc.Text[..200] + "..." : tc.Text;
                            AnsiConsole.MarkupLine($"  [grey]TextContent:[/] {Markup.Escape(preview ?? "(empty)")}");
                            break;
                        case FunctionCallContent fc:
                            AnsiConsole.MarkupLine($"  [grey]FunctionCallContent:[/] {Markup.Escape(fc.Name ?? "?")}({Markup.Escape(JsonSerializer.Serialize(fc.Arguments))})");
                            break;
                        case FunctionResultContent fr:
                            var resultPreview = fr.Result?.ToString();
                            resultPreview = resultPreview?.Length > 200 ? resultPreview[..200] + "..." : resultPreview;
                            AnsiConsole.MarkupLine($"  [grey]FunctionResultContent:[/] {Markup.Escape(resultPreview ?? "(empty)")}");
                            break;
                        default:
                            AnsiConsole.MarkupLine($"  [grey]{content.GetType().Name}[/]");
                            break;
                    }
                }
            }

            // Show metadata/additional properties
            if (msg.AdditionalProperties?.Count > 0)
            {
                AnsiConsole.MarkupLine($"  [grey]AdditionalProperties:[/] {Markup.Escape(JsonSerializer.Serialize(msg.AdditionalProperties))}");
            }

            AnsiConsole.WriteLine();
        }

        // Show what SQL Server columns you'd need
        AnsiConsole.MarkupLine("[bold yellow]--- SQL Server Schema Recommendation ---[/]");
        AnsiConsole.MarkupLine("""
        [grey]Based on the session structure, here are the fields you'd store:

        CREATE TABLE ChatMessages (
            Id              INT IDENTITY PRIMARY KEY,
            SessionId       NVARCHAR(100) NOT NULL,     -- links messages to a session
            UserId          NVARCHAR(100) NOT NULL,     -- links session to a user
            MessageIndex    INT NOT NULL,                -- ordering within session
            Role            NVARCHAR(20) NOT NULL,      -- 'user', 'assistant'
            TextContent     NVARCHAR(MAX),              -- the actual message text
            CreatedAt       DATETIME2 DEFAULT GETDATE(),

            INDEX IX_Session (SessionId, MessageIndex)
        );

        CREATE TABLE ChatSessions (
            SessionId       NVARCHAR(100) PRIMARY KEY,
            UserId          NVARCHAR(100) NOT NULL,
            CreatedAt       DATETIME2 DEFAULT GETDATE(),
            LastMessageAt   DATETIME2,

            INDEX IX_User (UserId)
        );
        [/]
        """);

        AnsiConsole.MarkupLine("[bold yellow]---------------------------------------[/]");
    }

    private static string ExtractSchemaSummary(string schemaJson)
    {
        try
        {
            var doc = JsonDocument.Parse(schemaJson);
            var root = doc.RootElement;

            JsonElement schema;
            if (root.TryGetProperty("data", out var data) && data.TryGetProperty("__schema", out schema))
            { }
            else if (root.TryGetProperty("__schema", out schema))
            { }
            else
            {
                return schemaJson;
            }

            var queryTypeName = schema.GetProperty("queryType").GetProperty("name").GetString();
            var mutationTypeName = schema.TryGetProperty("mutationType", out var mt) && mt.ValueKind != JsonValueKind.Null
                ? mt.GetProperty("name").GetString()
                : null;

            var types = schema.GetProperty("types");
            var sb = new StringBuilder();

            foreach (var type in types.EnumerateArray())
            {
                var typeName = type.GetProperty("name").GetString();
                if (typeName == null || typeName.StartsWith("__"))
                    continue;

                bool isQuery = typeName == queryTypeName;
                bool isMutation = typeName == mutationTypeName;

                if ((isQuery || isMutation) && type.TryGetProperty("fields", out var fields) && fields.ValueKind == JsonValueKind.Array)
                {
                    sb.AppendLine(isQuery ? "## Available Queries:" : "## Available Mutations:");

                    foreach (var field in fields.EnumerateArray())
                    {
                        var fieldName = field.GetProperty("name").GetString();
                        var description = field.TryGetProperty("description", out var desc) && desc.ValueKind == JsonValueKind.String
                            ? desc.GetString()
                            : null;

                        sb.Append($"- {fieldName}");
                        if (!string.IsNullOrEmpty(description))
                            sb.Append($": {description}");

                        if (field.TryGetProperty("args", out var args) && args.ValueKind == JsonValueKind.Array && args.GetArrayLength() > 0)
                        {
                            var argNames = new List<string>();
                            foreach (var arg in args.EnumerateArray())
                            {
                                var argName = arg.GetProperty("name").GetString();
                                if (argName != null)
                                    argNames.Add(argName);
                            }
                            if (argNames.Count > 0)
                                sb.Append($" (args: {string.Join(", ", argNames)})");
                        }

                        sb.AppendLine();
                    }
                    sb.AppendLine();
                }
            }

            var summary = sb.ToString();
            return string.IsNullOrWhiteSpace(summary) ? schemaJson : summary;
        }
        catch
        {
            return schemaJson;
        }
    }
}
