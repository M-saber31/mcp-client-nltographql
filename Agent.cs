using System.ComponentModel;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;

#pragma warning disable SKEXP0001

namespace Samples.Azure.Database.NL2SQL;

public class GraphQLAgent(McpClient mcpClient, ILogger logger)
{
    private readonly McpClient _mcpClient = mcpClient;
    private readonly ILogger _logger = logger;

    [KernelFunction("IntrospectSchema")]
    [Description("Retrieves the full GraphQL schema from the backend API. Call this to discover available types, fields, and their arguments before making queries.")]
    public async Task<string> IntrospectSchema()
    {
        _logger.LogInformation("Introspecting GraphQL schema via MCP server...");
        var result = await _mcpClient.CallToolAsync("introspect-schema");
        _logger.LogInformation("Schema introspection completed.");
        return result;
    }

    [KernelFunction("QueryGraphQL")]
    [Description("""
        Executes a GraphQL query against the backend API via the MCP server.
        The schema must have been introspected first (either at startup or by calling IntrospectSchema).
        The query must be a valid GraphQL query string.
        Example: query { dimension_stock_items(first: 10) { items { StockItemKey StockItem Color } } }
        """)]
    public async Task<string> QueryGraphQL(
        [Description("The GraphQL query string to execute")] string query,
        [Description("Optional JSON string of GraphQL variables, e.g. {\"id\": 1}")] string? variables = null)
    {
        _logger.LogInformation($"Executing GraphQL query: {query}");
        if (variables != null)
            _logger.LogInformation($"With variables: {variables}");

        var arguments = new Dictionary<string, object> { { "query", query } };

        if (!string.IsNullOrEmpty(variables))
        {
            var parsedVars = JsonSerializer.Deserialize<Dictionary<string, object>>(variables);
            if (parsedVars != null)
                arguments["variables"] = parsedVars;
        }

        var result = await _mcpClient.CallToolAsync("query-graphql", arguments);
        _logger.LogInformation("GraphQL query executed successfully.");
        return result;
    }
}