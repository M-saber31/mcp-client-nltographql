using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Samples.Azure.Database.NL2SQL;

public class McpClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _serverUrl;
    private string? _sessionId;
    private int _requestId = 0;
    private bool _initialized = false;

    public McpClient(string serverUrl)
    {
        _serverUrl = serverUrl.TrimEnd('/');
        _httpClient = new HttpClient(new HttpClientHandler
        {
            // Avoid proxy issues on corporate networks
            UseProxy = false
        });
        // Disable Expect: 100-continue which can cause issues with some servers
        _httpClient.DefaultRequestHeaders.ExpectContinue = false;
    }

    public async Task InitializeAsync()
    {
        var initRequest = new
        {
            jsonrpc = "2.0",
            id = NextId(),
            method = "initialize",
            @params = new
            {
                protocolVersion = "2025-03-26",
                capabilities = new { },
                clientInfo = new { name = "nl2graphql-client", version = "1.0" }
            }
        };

        try
        {
            await SendRequestAsync(initRequest);

            var notification = new
            {
                jsonrpc = "2.0",
                method = "notifications/initialized"
            };

            await SendNotificationAsync(notification);
            _initialized = true;
        }
        catch (HttpRequestException ex)
        {
            // If initialization fails (e.g. 405), try working without session.
            // Some MCP server configurations handle requests statelessly.
            Console.WriteLine($"MCP initialization handshake failed ({ex.Message}), will attempt stateless requests.");
            _sessionId = null;
            _initialized = false;
        }
    }

    public async Task<string> CallToolAsync(string toolName, Dictionary<string, object>? arguments = null)
    {
        // If we never initialized successfully, try a per-request init+call
        if (!_initialized)
        {
            return await CallToolWithInitAsync(toolName, arguments);
        }

        return await CallToolDirectAsync(toolName, arguments);
    }

    private async Task<string> CallToolDirectAsync(string toolName, Dictionary<string, object>? arguments)
    {
        var request = new
        {
            jsonrpc = "2.0",
            id = NextId(),
            method = "tools/call",
            @params = new
            {
                name = toolName,
                arguments = arguments ?? new Dictionary<string, object>()
            }
        };

        var response = await SendRequestAsync(request);
        return ExtractToolResult(response);
    }

    /// <summary>
    /// Sends initialize + initialized + tools/call as a fresh session per request.
    /// Used when the initial handshake failed, since the server creates a new
    /// transport per request and may not persist session state.
    /// </summary>
    private async Task<string> CallToolWithInitAsync(string toolName, Dictionary<string, object>? arguments)
    {
        // Step 1: Initialize a fresh session
        _sessionId = null;
        var initRequest = new
        {
            jsonrpc = "2.0",
            id = NextId(),
            method = "initialize",
            @params = new
            {
                protocolVersion = "2025-03-26",
                capabilities = new { },
                clientInfo = new { name = "nl2graphql-client", version = "1.0" }
            }
        };

        await SendRequestAsync(initRequest);

        var notification = new
        {
            jsonrpc = "2.0",
            method = "notifications/initialized"
        };
        await SendNotificationAsync(notification);

        // Step 2: Call the tool using the session from init
        return await CallToolDirectAsync(toolName, arguments);
    }

    private static string ExtractToolResult(JsonElement response)
    {
        if (response.TryGetProperty("result", out var result) &&
            result.TryGetProperty("content", out var content) &&
            content.ValueKind == JsonValueKind.Array &&
            content.GetArrayLength() > 0)
        {
            var firstContent = content[0];
            if (firstContent.TryGetProperty("text", out var text))
            {
                return text.GetString() ?? string.Empty;
            }
        }

        if (response.TryGetProperty("error", out var error))
        {
            throw new InvalidOperationException($"MCP error: {error.GetRawText()}");
        }

        return response.GetRawText();
    }

    private int NextId() => ++_requestId;

    private async Task<JsonElement> SendRequestAsync(object request)
    {
        var json = JsonSerializer.Serialize(request);

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, _serverUrl);
        httpRequest.Content = new StringContent(json, Encoding.UTF8, "application/json");
        httpRequest.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        httpRequest.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        httpRequest.Version = new Version(1, 1);

        if (_sessionId != null)
            httpRequest.Headers.Add("mcp-session-id", _sessionId);

        var response = await _httpClient.SendAsync(httpRequest);
        response.EnsureSuccessStatusCode();

        // Capture session ID from response headers
        if (response.Headers.TryGetValues("mcp-session-id", out var sessionIds))
            _sessionId = sessionIds.FirstOrDefault();

        var contentType = response.Content.Headers.ContentType?.MediaType;
        var responseBody = await response.Content.ReadAsStringAsync();

        if (contentType == "text/event-stream")
        {
            return ParseSseResponse(responseBody);
        }

        return JsonSerializer.Deserialize<JsonElement>(responseBody);
    }

    private async Task SendNotificationAsync(object notification)
    {
        var json = JsonSerializer.Serialize(notification);

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, _serverUrl);
        httpRequest.Content = new StringContent(json, Encoding.UTF8, "application/json");
        httpRequest.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        httpRequest.Version = new Version(1, 1);

        if (_sessionId != null)
            httpRequest.Headers.Add("mcp-session-id", _sessionId);

        var response = await _httpClient.SendAsync(httpRequest);
        // Notifications return 202 Accepted - don't throw on non-200
        if (response.Headers.TryGetValues("mcp-session-id", out var sessionIds))
            _sessionId = sessionIds.FirstOrDefault();
    }

    private static JsonElement ParseSseResponse(string sseBody)
    {
        var lines = sseBody.Split('\n');
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.StartsWith("data:"))
            {
                var data = trimmed[5..].Trim();
                if (!string.IsNullOrEmpty(data))
                {
                    try
                    {
                        return JsonSerializer.Deserialize<JsonElement>(data);
                    }
                    catch (JsonException)
                    {
                        continue;
                    }
                }
            }
        }

        throw new InvalidOperationException("No valid JSON data found in SSE response");
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}