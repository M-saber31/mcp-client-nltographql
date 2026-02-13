using System.Text;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using DotNetEnv;
using Spectre.Console;
using Azure.Identity;

#pragma warning disable SKEXP0001, SKEXP0010, SKEXP0020

namespace Samples.Azure.Database.NL2SQL;

public class ChatBot
{
    const string VERSION = "2.0";
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

        using var mcpClient = new McpClient(mcpServerUrl);

        var openAIPromptExecutionSettings = new AzureOpenAIPromptExecutionSettings()
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };

        (var logger, var kernel, var ai, var schemaSummary) = await AnsiConsole.Status().StartAsync("Booting up agents...", async ctx =>
        {
            AnsiConsole.WriteLine("Initializing orchestrator agent...");
            ctx.Spinner(Spinner.Known.Default);
            ctx.SpinnerStyle(Style.Parse("yellow"));

            AnsiConsole.WriteLine("Initializing kernel...");
            var credentials = new DefaultAzureCredential();
            var sc = new ServiceCollection();

            if (string.IsNullOrEmpty(azureOpenAIApiKey))
            {
                sc.AddAzureOpenAIChatCompletion(chatModelDeploymentName, azureOpenAIEndpoint, credentials);
            }
            else
            {
                sc.AddAzureOpenAIChatCompletion(chatModelDeploymentName, azureOpenAIEndpoint, azureOpenAIApiKey);
            }

            sc.AddKernel();
            sc.AddLogging(b =>
            {
                b.ClearProviders();
                b.SetMinimumLevel(enableDebug ? LogLevel.Debug : LogLevel.None);
                b.AddProvider(new SpectreConsoleLoggerProvider());
            });

            var services = sc.BuildServiceProvider();
            var loggerFactory = services.GetRequiredService<ILoggerFactory>();

            var orchestratorAgentLogger = loggerFactory.CreateLogger("OrchestratorAgent");
            var specializedAgentLogger = loggerFactory.CreateLogger("SpecializedAgent");

            AnsiConsole.WriteLine("Connecting to MCP server...");
            AnsiConsole.WriteLine($"MCP Server URL: {mcpServerUrl}");
            await mcpClient.InitializeAsync();
            AnsiConsole.WriteLine("MCP session established.");

            AnsiConsole.WriteLine("Introspecting GraphQL schema...");
            var schemaJson = await mcpClient.CallToolAsync("introspect-schema");
            var schemaSummary = ExtractSchemaSummary(schemaJson);
            AnsiConsole.WriteLine("Schema introspection completed.");

            AnsiConsole.WriteLine("Initializing specialized agents...");
            var kernel = services.GetRequiredService<Kernel>();
            kernel.Plugins.AddFromObject(new GraphQLAgent(mcpClient, specializedAgentLogger));

            foreach (var p in kernel.Plugins)
            {
                foreach (var f in p.GetFunctionsMetadata())
                {
                    AnsiConsole.WriteLine($"Agent: {p.Name}, Tool: {f.Name}");
                }
            }
            var ai = kernel.GetRequiredService<IChatCompletionService>();

            AnsiConsole.WriteLine("All done.");

            return (orchestratorAgentLogger, kernel, ai, schemaSummary);
        });

        AnsiConsole.WriteLine("Ready to chat! Hit 'ctrl-c' to quit.");
        var chat = new ChatHistory($"""
            You are an AI assistant that helps users query data via a GraphQL API.
            The API exposes the following queries:
 
            {schemaSummary}
 
            Use a professional tone when answering and provide a summary of data instead of lists.
            If users ask about topics you don't know, answer that you don't know. Today's date is {DateTime.Now:yyyy-MM-dd}.
            You must use the provided QueryGraphQL tool to query the GraphQL API with valid GraphQL query strings.
            If you need more details about the schema structure or available fields, use the IntrospectSchema tool.
            If the request is complex, break it down into smaller steps and call the QueryGraphQL tool as many times as needed.
        """);
        var builder = new StringBuilder();
        long sessionPromptTokens = 0;
        long sessionCompletionTokens = 0;
        long sessionTotalTokens = 0;
        int requestCount = 0;

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
                    chat.RemoveRange(1, chat.Count - 1);
                    sessionPromptTokens = 0;
                    sessionCompletionTokens = 0;
                    sessionTotalTokens = 0;
                    requestCount = 0;
                    AnsiConsole.WriteLine("Chat history and token counters cleared.");
                    continue;
                case "/h":
                    foreach (var message in chat)
                    {
                        AnsiConsole.WriteLine($"> ---------- {message.Role} ----------");
                        AnsiConsole.WriteLine($"> MESSAGE  > {message.Content}");
                        AnsiConsole.WriteLine($"> METADATA > {JsonSerializer.Serialize(message.Metadata)}");
                        AnsiConsole.WriteLine($"> ------------------------------------");
                    }
                    continue;
                case "/tokens":
                    var historyChars = chat.Sum(m => m.Content?.Length ?? 0);
                    var estimatedHistoryTokens = historyChars / 4;
                    AnsiConsole.MarkupLine($"[cyan]📊 Token Usage Report[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Requests made:          {requestCount}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Session prompt tokens:   {sessionPromptTokens:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Session completion tokens:{sessionCompletionTokens:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Session total tokens:    {sessionTotalTokens:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Chat history messages:   {chat.Count}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Chat history chars:      {historyChars:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Est. history tokens:     ~{estimatedHistoryTokens:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Rate limit (S0 tier):    200,000 TPM[/]");
                    continue;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.WriteLine("🤖: Formulating answer...");
            builder.Clear();
            chat.AddUserMessage(question);
            var firstLine = true;
            var maxRetries = 5;
            var succeeded = false;
            long reqPromptTokens = 0;
            long reqCompletionTokens = 0;
            long reqTotalTokens = 0;

            for (int attempt = 0; attempt < maxRetries && !succeeded; attempt++)
            {
                try
                {
                    if (attempt > 0)
                    {
                        AnsiConsole.MarkupLine($"[yellow]🔄 Retry attempt {attempt}/{maxRetries - 1}...[/]");
                        firstLine = true;
                        builder.Clear();
                    }

                    await foreach (var message in ai.GetStreamingChatMessageContentsAsync(chat, openAIPromptExecutionSettings, kernel))
                    {
                        if (!enableDebug)
                            if (firstLine && message.Content != null && message.Content.Length > 0)
                            {
                                AnsiConsole.Cursor.MoveUp();
                                AnsiConsole.WriteLine("                                  ");
                                AnsiConsole.Cursor.MoveUp();
                                AnsiConsole.Write($"🤖: ");
                                firstLine = false;
                            }
                        AnsiConsole.Write(message.Content ?? string.Empty);
                        builder.Append(message.Content);

                        if (message.Metadata != null)
                        {
                            if (message.Metadata.TryGetValue("Usage", out var usage) && usage != null)
                            {
                                var usageJson = JsonSerializer.Serialize(usage);
                                var usageDoc = JsonDocument.Parse(usageJson);
                                var root = usageDoc.RootElement;
                                if (root.TryGetProperty("PromptTokens", out var pt))
                                    reqPromptTokens = pt.GetInt64();
                                if (root.TryGetProperty("CompletionTokens", out var ct))
                                    reqCompletionTokens = ct.GetInt64();
                                if (root.TryGetProperty("TotalTokens", out var tt))
                                    reqTotalTokens = tt.GetInt64();
                            }
                        }
                    }
                    succeeded = true;
                }
                catch (HttpOperationException ex) when (ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                {
                    var waitSeconds = (int)Math.Pow(2, attempt) * 15;
                    AnsiConsole.MarkupLine($"[yellow]⚠️ Rate limit hit. Waiting {waitSeconds} seconds before retrying...[/]");
                    await Task.Delay(TimeSpan.FromSeconds(waitSeconds));
                }
            }

            if (!succeeded)
            {
                AnsiConsole.MarkupLine("[red]❌ Failed after all retry attempts. Try again later.[/]");
                chat.RemoveAt(chat.Count - 1);
            }
            else
            {
                AnsiConsole.WriteLine();
                chat.AddAssistantMessage(builder.ToString());

                requestCount++;
                sessionPromptTokens += reqPromptTokens;
                sessionCompletionTokens += reqCompletionTokens;
                sessionTotalTokens += reqTotalTokens;

                AnsiConsole.MarkupLine($"[grey]📊 This request: prompt={reqPromptTokens:N0} completion={reqCompletionTokens:N0} total={reqTotalTokens:N0} | Session total: {sessionTotalTokens:N0}/200,000 TPM[/]");
            }
        }
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