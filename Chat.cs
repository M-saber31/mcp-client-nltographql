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
using System.ClientModel;

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
            AnsiConsole.WriteLine("Schema introspection completed.");

            AnsiConsole.WriteLine("Initializing specialized agents...");
            var kernel = services.GetRequiredService<Kernel>();
            var graphQLAgent = new GraphQLAgent(mcpClient, specializedAgentLogger);
            graphQLAgent.LoadSchema(schemaJson);
            var schemaSummary = graphQLAgent.GetCompactSummary();
            kernel.Plugins.AddFromObject(graphQLAgent);

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
            {schemaSummary}
            IMPORTANT workflow:
            1. Use ListQueries or SearchSchema to find relevant queries.
            2. Use DescribeQuery to get the arguments and return type of a query.
            3. Use DescribeType to explore the fields of a return type so you know what to select.
            4. Use QueryGraphQL to execute the query.
            Use a professional tone and provide summaries instead of raw lists.
            If users ask about topics you don't know, answer that you don't know. Today's date is {DateTime.Now:yyyy-MM-dd}.
        """);
        var builder = new StringBuilder();
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
                    var estimatedNextPrompt = historyChars / 4;
                    AnsiConsole.MarkupLine($"[cyan]📊 Token Usage Report[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Requests made:             {requestCount}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Session total tokens used:  ≈{sessionTotalTokens:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Chat history messages:      {chat.Count}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Chat history chars:         {historyChars:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Est. next prompt tokens:    ≈{estimatedNextPrompt:N0}[/]");
                    AnsiConsole.MarkupLine($"[cyan]   Rate limit (S0 tier):       200,000 TPM[/]");
                    if (estimatedNextPrompt > 150_000)
                        AnsiConsole.MarkupLine($"[red]   ⚠️ WARNING: Next prompt is close to rate limit! Use /ch to clear history.[/]");
                    else if (estimatedNextPrompt > 100_000)
                        AnsiConsole.MarkupLine($"[yellow]   ⚠️ CAUTION: History is getting large. Consider using /ch to clear.[/]");
                    continue;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.WriteLine("🤖: Formulating answer...");
            builder.Clear();
            chat.AddUserMessage(question);
            var firstLine = true;
            var maxRetries = 5;
            var succeeded = false;

            // Estimate prompt tokens from entire chat history (sent with every request)
            var promptChars = chat.Sum(m => m.Content?.Length ?? 0);
            var estPromptTokens = promptChars / 4;

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
                    }
                    succeeded = true;
                }
                catch (ClientResultException ex) when (ex.Status == 429)
                {
                    var waitSeconds = (int)Math.Pow(2, attempt) * 15;
                    AnsiConsole.MarkupLine($"[yellow]⚠️ Rate limit hit (429). Waiting {waitSeconds} seconds before retrying...[/]");
                    await Task.Delay(TimeSpan.FromSeconds(waitSeconds));
                }
                catch (HttpOperationException ex) when (ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                {
                    var waitSeconds = (int)Math.Pow(2, attempt) * 15;
                    AnsiConsole.MarkupLine($"[yellow]⚠️ Rate limit hit (429). Waiting {waitSeconds} seconds before retrying...[/]");
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
                var estCompletionTokens = builder.Length / 4;
                var estTotalTokens = estPromptTokens + estCompletionTokens;
                sessionTotalTokens += estTotalTokens;

                AnsiConsole.MarkupLine($"[grey]📊 Est. tokens: prompt≈{estPromptTokens:N0} completion≈{estCompletionTokens:N0} total≈{estTotalTokens:N0} | Session total≈{sessionTotalTokens:N0}/200,000 TPM[/]");
            }
        }
    }

}