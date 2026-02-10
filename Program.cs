using DotNetEnv;
using Samples.Azure.Database.NL2SQL;
using Samples.Azure.Database.NL2SQL.Endpoints;
using Samples.Azure.Database.NL2SQL.Helpers;
using Samples.Azure.Database.NL2SQL.Hubs;
using Samples.Azure.Database.NL2SQL.Services;

Env.Load(".env");

var openAiUrl = Environment.GetEnvironmentVariable("OPENAI_URL") ?? throw new InvalidOperationException("OPENAI_URL not set");
var openAiKey = Environment.GetEnvironmentVariable("OPENAI_KEY") ?? string.Empty;
var chatDeployment = Environment.GetEnvironmentVariable("OPENAI_CHAT_DEPLOYMENT_NAME") ?? throw new InvalidOperationException("OPENAI_CHAT_DEPLOYMENT_NAME not set");
var mcpServerUrl = Environment.GetEnvironmentVariable("MCP_SERVER_URL") ?? throw new InvalidOperationException("MCP_SERVER_URL not set");

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

var mcpClient = new McpClient(mcpServerUrl);
builder.Services.AddSingleton(mcpClient);

builder.Services.AddSingleton<IKernelFactory>(sp =>
    new KernelFactory(
        sp.GetRequiredService<McpClient>(),
        sp.GetRequiredService<ILoggerFactory>(),
        openAiUrl,
        openAiKey,
        chatDeployment));

builder.Services.AddSingleton<ChatSessionManager>();

builder.Services.AddSignalR();

builder.Services.AddCors(options =>
{
    options.AddPolicy("NextJs", policy =>
    {
        policy.WithOrigins("http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

var app = builder.Build();

app.UseCors("NextJs");

var logger = app.Services.GetRequiredService<ILogger<Program>>();

logger.LogInformation("Initializing MCP client...");
await mcpClient.InitializeAsync();

logger.LogInformation("Introspecting GraphQL schema...");
var schemaJson = await mcpClient.CallToolAsync("introspect-schema");
var schemaSummary = SchemaHelper.ExtractSchemaSummary(schemaJson);
logger.LogInformation("Schema introspection completed.");

var sessionManager = app.Services.GetRequiredService<ChatSessionManager>();
sessionManager.SetSchemaSummary(schemaSummary);

app.MapHub<ChatHub>("/chat-hub");
app.MapChatEndpoints();

logger.LogInformation("Server started. Ready to accept connections.");

app.Run();
