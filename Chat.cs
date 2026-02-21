using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
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

        (var agent, var plannerAgent, var mcpClient, var session) = await AnsiConsole.Status().StartAsync("Booting up agents...", async ctx =>
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

            // Load allowed queries from JSON file
            AnsiConsole.WriteLine("Loading allowed queries...");
            var allowedQueriesJson = await File.ReadAllTextAsync("allowed-queries.json");
            var allowedQueries = JsonSerializer.Deserialize<List<AllowedQuery>>(allowedQueriesJson,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? throw new InvalidOperationException("Failed to load allowed-queries.json");
            var queryCatalog = BuildQueryCatalog(allowedQueries);
            AnsiConsole.MarkupLine($"[yellow]Loaded {allowedQueries.Count} allowed queries. ({sw.Elapsed.TotalSeconds:F2}s)[/]");
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
                    Today's date is {DateTime.Now:yyyy-MM-dd}.

                    ## ALLOWED QUERIES
                    You may ONLY execute the following pre-approved GraphQL queries using the query-graphql tool.
                    Do NOT invent, modify, or construct any other queries. Use ONLY the exact query templates below,
                    substituting variable values as needed based on the user's request.

                    {queryCatalog}

                    ## RULES
                    - You MUST use one of the allowed queries above. If the user asks for data that none of these queries can answer, politely explain that the query is not available.
                    - Always set the $first variable to limit results (e.g. 20) unless the user specifically asks for more.
                    - Use a professional tone and provide a summary of data instead of raw lists.
                    - If the user's request maps to multiple allowed queries, you may call query-graphql multiple times.

                    ## SCIENTIFIC / KNOWLEDGE QUESTIONS
                    - When the user asks a scientific question, a knowledge question, or any question that could be answered by course content, you MUST use the contents query to search for relevant course material.
                    - Extract the main topic or keyword from the user's question and use it as the $Topic variable in the contents query.
                    - Your answer MUST be based ONLY on what the contents query returns. Do NOT use your own knowledge to answer.
                    - If the contents query returns empty results (no items), you MUST respond with exactly: "There are no course content for the question you are asking"
                    - Do NOT make up, fabricate, or supplement answers with information that was not returned by the query.
                    """,
                name: "GraphQLAssistant",
                tools: mcpTools
                    .Where(t => t.Name == "query-graphql")
                    .Select(t => (AITool)new TimedAIFunction(t))
                    .ToList(),
                clientFactory: inner => new TimingChatClient(inner),
                loggerFactory: loggerFactory
            );

            AnsiConsole.MarkupLine($"[yellow]Agent initialized. ({sw.Elapsed.TotalSeconds:F2}s)[/]");
            sw.Restart();

            // Create the planner agent — pure reasoning, no MCP tools.
            // It receives structured course data and produces a study plan.
            AnsiConsole.WriteLine("Initializing planner agent...");
            var plannerAgent = chatClient.AsAIAgent(
                instructions: """
                    You are a student study advisor.
                    You will receive a JSON summary of a student's unfinished course content
                    (subjects, learning levels, and content titles).

                    Create a structured weekly study plan that:
                    - Groups related content by subject and learning level
                    - Prioritizes subjects with lower completion percentages first
                    - Suggests a realistic number of content items per day (2-3 items max)
                    - Uses clear headings (Week 1, Day 1: ..., etc.)
                    - Is encouraging and supportive in tone

                    Output the study plan as plain readable text only.
                    Do not call any tools or APIs.
                    """,
                name: "StudyPlanner",
                clientFactory: inner => new TimingChatClient(inner),
                loggerFactory: loggerFactory
            );
            AnsiConsole.MarkupLine($"[yellow]Planner agent initialized. ({sw.Elapsed.TotalSeconds:F2}s)[/]");

            // Create a session to manage conversation state
            var session = await agent.CreateSessionAsync();

            AnsiConsole.WriteLine("All done.");
            return (agent, plannerAgent, mcpClient, session);
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

                // Detect study plan trigger:
                // Matches messages like "please create a study plan for student with username alice"
                if (question.Contains("study plan", StringComparison.OrdinalIgnoreCase))
                {
                    var usernameMatch = Regex.Match(
                        question,
                        @"username\s+[""']?(\w+)[""']?",
                        RegexOptions.IgnoreCase);

                    if (usernameMatch.Success)
                    {
                        var username = usernameMatch.Groups[1].Value;
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine($"[yellow]>> Study plan workflow triggered for student '{username}'[/]");
                        AnsiConsole.MarkupLine("[yellow]>> Running workflow: DataGathering → PlannerAgent...[/]");
                        AnsiConsole.WriteLine();

                        var planSw = Stopwatch.StartNew();
                        var plan = await StudyPlanWorkflowRunner.RunAsync(
                            username: username,
                            userId: username,   // adjust if userId differs from username
                            graphqlAgent: agent,
                            plannerAgent: plannerAgent);
                        planSw.Stop();

                        AnsiConsole.Write("🤖: ");
                        Console.WriteLine(plan);
                        AnsiConsole.MarkupLine($"[yellow]>> Study plan generated in {planSw.Elapsed.TotalSeconds:F2}s[/]");
                        continue;
                    }
                }

                AnsiConsole.WriteLine();

                var turnSw = Stopwatch.StartNew();
                AnsiConsole.Write("🤖: ");

                await foreach (var update in agent.RunStreamingAsync(question, session))
                {
                    var text = update.Text;
                    if (!string.IsNullOrEmpty(text))
                    {
                        Console.Write(text);
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

    private static string BuildQueryCatalog(List<AllowedQuery> queries)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < queries.Count; i++)
        {
            sb.AppendLine($"### Query {i + 1}: {queries[i].Description}");
            sb.AppendLine($"```graphql");
            sb.AppendLine(queries[i].Query);
            sb.AppendLine($"```");
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

public class AllowedQuery
{
    public string Description { get; set; } = string.Empty;
    public string Query { get; set; } = string.Empty;
}
