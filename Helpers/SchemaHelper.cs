using System.Text;
using System.Text.Json;

namespace Samples.Azure.Database.NL2SQL.Helpers;

public static class SchemaHelper
{
    public static string ExtractSchemaSummary(string schemaJson)
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
