using System.ComponentModel;
using System.Text;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;

#pragma warning disable SKEXP0001

namespace Samples.Azure.Database.NL2SQL;

public class GraphQLAgent(McpClient mcpClient, ILogger logger)
{
    private readonly McpClient _mcpClient = mcpClient;
    private readonly ILogger _logger = logger;

    // In-memory parsed schema for on-demand exploration
    private List<SchemaField> _queries = [];
    private List<SchemaField> _mutations = [];
    private Dictionary<string, SchemaType> _types = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Parses the raw introspection JSON and stores it in memory for tool-based exploration.
    /// Call this once at startup.
    /// </summary>
    public void LoadSchema(string schemaJson)
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
                _logger.LogWarning("Could not find __schema in introspection result.");
                return;
            }

            var queryTypeName = schema.GetProperty("queryType").GetProperty("name").GetString();
            var mutationTypeName = schema.TryGetProperty("mutationType", out var mt) && mt.ValueKind != JsonValueKind.Null
                ? mt.GetProperty("name").GetString()
                : null;

            var types = schema.GetProperty("types");

            foreach (var type in types.EnumerateArray())
            {
                var typeName = type.GetProperty("name").GetString();
                if (typeName == null || typeName.StartsWith("__"))
                    continue;

                // Parse fields for this type
                var fields = new List<TypeField>();
                if (type.TryGetProperty("fields", out var fieldsEl) && fieldsEl.ValueKind == JsonValueKind.Array)
                {
                    foreach (var f in fieldsEl.EnumerateArray())
                    {
                        fields.Add(new TypeField
                        {
                            Name = f.GetProperty("name").GetString() ?? "",
                            TypeName = FlattenType(f.GetProperty("type")),
                            Description = f.TryGetProperty("description", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null
                        });
                    }
                }

                // Parse enum values
                var enumValues = new List<string>();
                if (type.TryGetProperty("enumValues", out var enumEl) && enumEl.ValueKind == JsonValueKind.Array)
                {
                    foreach (var ev in enumEl.EnumerateArray())
                        enumValues.Add(ev.GetProperty("name").GetString() ?? "");
                }

                var kind = type.TryGetProperty("kind", out var k) ? k.GetString() : null;

                _types[typeName] = new SchemaType
                {
                    Name = typeName,
                    Kind = kind,
                    Description = type.TryGetProperty("description", out var td) && td.ValueKind == JsonValueKind.String ? td.GetString() : null,
                    Fields = fields,
                    EnumValues = enumValues
                };

                // Extract top-level queries and mutations
                bool isQuery = typeName == queryTypeName;
                bool isMutation = typeName == mutationTypeName;

                if ((isQuery || isMutation) && fieldsEl.ValueKind == JsonValueKind.Array)
                {
                    var targetList = isQuery ? _queries : _mutations;

                    foreach (var field in fieldsEl.EnumerateArray())
                    {
                        var args = new List<SchemaArg>();
                        if (field.TryGetProperty("args", out var argsEl) && argsEl.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var arg in argsEl.EnumerateArray())
                            {
                                args.Add(new SchemaArg
                                {
                                    Name = arg.GetProperty("name").GetString() ?? "",
                                    TypeName = FlattenType(arg.GetProperty("type")),
                                    Description = arg.TryGetProperty("description", out var ad) && ad.ValueKind == JsonValueKind.String ? ad.GetString() : null
                                });
                            }
                        }

                        targetList.Add(new SchemaField
                        {
                            Name = field.GetProperty("name").GetString() ?? "",
                            Description = field.TryGetProperty("description", out var fd) && fd.ValueKind == JsonValueKind.String ? fd.GetString() : null,
                            ReturnType = FlattenType(field.GetProperty("type")),
                            Args = args
                        });
                    }
                }
            }

            _logger.LogInformation($"Schema loaded: {_queries.Count} queries, {_mutations.Count} mutations, {_types.Count} types.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to parse schema.");
        }
    }

    /// <summary>
    /// Returns a compact list of query/mutation names for the system prompt.
    /// </summary>
    public string GetCompactSummary()
    {
        var sb = new StringBuilder();
        if (_queries.Count > 0)
        {
            sb.AppendLine("Available queries: " + string.Join(", ", _queries.Select(q => q.Name)));
        }
        if (_mutations.Count > 0)
        {
            sb.AppendLine("Available mutations: " + string.Join(", ", _mutations.Select(m => m.Name)));
        }
        return sb.ToString();
    }

    [KernelFunction("ListQueries")]
    [Description("Lists all available GraphQL queries with their descriptions. Call this first to discover what data you can query.")]
    public string ListQueries()
    {
        _logger.LogInformation("ListQueries called.");
        if (_queries.Count == 0) return "No queries available.";

        var sb = new StringBuilder();
        sb.AppendLine("Available GraphQL Queries:");
        foreach (var q in _queries)
        {
            sb.Append($"- {q.Name}");
            if (!string.IsNullOrEmpty(q.Description))
                sb.Append($": {q.Description}");
            sb.Append($" -> {q.ReturnType}");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    [KernelFunction("ListMutations")]
    [Description("Lists all available GraphQL mutations with their descriptions.")]
    public string ListMutations()
    {
        _logger.LogInformation("ListMutations called.");
        if (_mutations.Count == 0) return "No mutations available.";

        var sb = new StringBuilder();
        sb.AppendLine("Available GraphQL Mutations:");
        foreach (var m in _mutations)
        {
            sb.Append($"- {m.Name}");
            if (!string.IsNullOrEmpty(m.Description))
                sb.Append($": {m.Description}");
            sb.Append($" -> {m.ReturnType}");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    [KernelFunction("DescribeQuery")]
    [Description("Gets the full details of a specific GraphQL query including its arguments and return type. Use this after ListQueries to understand how to call a specific query.")]
    public string DescribeQuery(
        [Description("The exact name of the query to describe")] string queryName)
    {
        _logger.LogInformation($"DescribeQuery called for: {queryName}");
        var field = _queries.FirstOrDefault(q => q.Name.Equals(queryName, StringComparison.OrdinalIgnoreCase));
        if (field == null)
            return $"Query '{queryName}' not found. Use ListQueries to see available queries.";

        return FormatFieldDetails(field, "Query");
    }

    [KernelFunction("DescribeMutation")]
    [Description("Gets the full details of a specific GraphQL mutation including its arguments and return type.")]
    public string DescribeMutation(
        [Description("The exact name of the mutation to describe")] string mutationName)
    {
        _logger.LogInformation($"DescribeMutation called for: {mutationName}");
        var field = _mutations.FirstOrDefault(m => m.Name.Equals(mutationName, StringComparison.OrdinalIgnoreCase));
        if (field == null)
            return $"Mutation '{mutationName}' not found. Use ListMutations to see available mutations.";

        return FormatFieldDetails(field, "Mutation");
    }

    [KernelFunction("DescribeType")]
    [Description("Gets the fields and structure of a specific GraphQL type. Use this to understand what fields are available on a return type so you can select them in your query.")]
    public string DescribeType(
        [Description("The exact name of the GraphQL type to describe")] string typeName)
    {
        _logger.LogInformation($"DescribeType called for: {typeName}");
        if (!_types.TryGetValue(typeName, out var schemaType))
            return $"Type '{typeName}' not found. Use SearchSchema to find type names.";

        var sb = new StringBuilder();
        sb.AppendLine($"Type: {schemaType.Name} ({schemaType.Kind})");
        if (!string.IsNullOrEmpty(schemaType.Description))
            sb.AppendLine($"Description: {schemaType.Description}");

        if (schemaType.EnumValues.Count > 0)
        {
            sb.AppendLine($"Enum values: {string.Join(", ", schemaType.EnumValues)}");
        }

        if (schemaType.Fields.Count > 0)
        {
            sb.AppendLine("Fields:");
            foreach (var f in schemaType.Fields)
            {
                sb.Append($"  - {f.Name}: {f.TypeName}");
                if (!string.IsNullOrEmpty(f.Description))
                    sb.Append($" ({f.Description})");
                sb.AppendLine();
            }
        }
        return sb.ToString();
    }

    [KernelFunction("SearchSchema")]
    [Description("Searches across all query names, type names, and field names for a keyword. Use this when you're not sure which query or type to use.")]
    public string SearchSchema(
        [Description("The keyword to search for (case-insensitive)")] string keyword)
    {
        _logger.LogInformation($"SearchSchema called for: {keyword}");
        var sb = new StringBuilder();
        var kw = keyword.ToLowerInvariant();

        var matchingQueries = _queries.Where(q => q.Name.Contains(kw, StringComparison.OrdinalIgnoreCase)).ToList();
        if (matchingQueries.Count > 0)
        {
            sb.AppendLine("Matching queries:");
            foreach (var q in matchingQueries)
                sb.AppendLine($"  - {q.Name}: {q.Description ?? "(no description)"} -> {q.ReturnType}");
        }

        var matchingMutations = _mutations.Where(m => m.Name.Contains(kw, StringComparison.OrdinalIgnoreCase)).ToList();
        if (matchingMutations.Count > 0)
        {
            sb.AppendLine("Matching mutations:");
            foreach (var m in matchingMutations)
                sb.AppendLine($"  - {m.Name}: {m.Description ?? "(no description)"} -> {m.ReturnType}");
        }

        var matchingTypes = _types.Values
            .Where(t => t.Name.Contains(kw, StringComparison.OrdinalIgnoreCase)
                     || t.Fields.Any(f => f.Name.Contains(kw, StringComparison.OrdinalIgnoreCase)))
            .Take(20)
            .ToList();
        if (matchingTypes.Count > 0)
        {
            sb.AppendLine("Matching types:");
            foreach (var t in matchingTypes)
            {
                sb.Append($"  - {t.Name} ({t.Kind})");
                var matchingFields = t.Fields.Where(f => f.Name.Contains(kw, StringComparison.OrdinalIgnoreCase)).ToList();
                if (matchingFields.Count > 0)
                    sb.Append($" [matching fields: {string.Join(", ", matchingFields.Select(f => f.Name))}]");
                sb.AppendLine();
            }
        }

        return sb.Length == 0 ? $"No results found for '{keyword}'." : sb.ToString();
    }

    [KernelFunction("IntrospectSchema")]
    [Description("Retrieves the full raw GraphQL schema from the backend API. Only use this if the other schema tools don't provide enough detail.")]
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
        Use ListQueries and DescribeQuery first to understand the schema, then construct a valid GraphQL query.
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

    private static string FormatFieldDetails(SchemaField field, string kind)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{kind}: {field.Name}");
        if (!string.IsNullOrEmpty(field.Description))
            sb.AppendLine($"Description: {field.Description}");
        sb.AppendLine($"Returns: {field.ReturnType}");

        if (field.Args.Count > 0)
        {
            sb.AppendLine("Arguments:");
            foreach (var arg in field.Args)
            {
                sb.Append($"  - {arg.Name}: {arg.TypeName}");
                if (!string.IsNullOrEmpty(arg.Description))
                    sb.Append($" ({arg.Description})");
                sb.AppendLine();
            }
        }
        else
        {
            sb.AppendLine("Arguments: none");
        }
        return sb.ToString();
    }

    private static string FlattenType(JsonElement typeEl)
    {
        var kind = typeEl.GetProperty("kind").GetString();
        switch (kind)
        {
            case "NON_NULL":
                return FlattenType(typeEl.GetProperty("ofType")) + "!";
            case "LIST":
                return "[" + FlattenType(typeEl.GetProperty("ofType")) + "]";
            default:
                return typeEl.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String
                    ? n.GetString() ?? "Unknown"
                    : "Unknown";
        }
    }

    // Internal data models
    private class SchemaField
    {
        public string Name { get; set; } = "";
        public string? Description { get; set; }
        public string ReturnType { get; set; } = "";
        public List<SchemaArg> Args { get; set; } = [];
    }

    private class SchemaArg
    {
        public string Name { get; set; } = "";
        public string TypeName { get; set; } = "";
        public string? Description { get; set; }
    }

    private class SchemaType
    {
        public string Name { get; set; } = "";
        public string? Kind { get; set; }
        public string? Description { get; set; }
        public List<TypeField> Fields { get; set; } = [];
        public List<string> EnumValues { get; set; } = [];
    }

    private class TypeField
    {
        public string Name { get; set; } = "";
        public string TypeName { get; set; } = "";
        public string? Description { get; set; }
    }
}
