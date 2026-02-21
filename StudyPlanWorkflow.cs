using System.Text;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

namespace Samples.Azure.Database.NL2SQL;

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

/// <summary>
/// Intermediate result passed from DataGatheringExecutor to the planner/all-done executors.
/// </summary>
public record UnfinishedContentSummary(
    string Username,
    bool HasContent,
    string DataJson   // structured JSON returned by the GraphQL agent
);

/// <summary>Helper DTO serialized into the initial ChatMessage.</summary>
public record StudyPlanMessage(string Username, string UserId);

// ---------------------------------------------------------------------------
// Executor 1: Gathers all unfinished course data using the existing GraphQL agent
//
// Inherits Executor<TInput, TOutput> — the typed base class that implements
// ConfigureRoutes automatically (no source generator needed).
// ---------------------------------------------------------------------------

/// <summary>
/// Receives a ChatMessage containing the student's username and userId,
/// runs the three GraphQL queries via the existing GraphQL agent (in a fresh session),
/// and returns a structured summary of unfinished content.
/// </summary>
internal sealed class DataGatheringExecutor : Executor<ChatMessage, UnfinishedContentSummary>
{
    private readonly AIAgent _graphqlAgent;

    public DataGatheringExecutor(AIAgent graphqlAgent) : base("DataGathering")
        => _graphqlAgent = graphqlAgent;

    // Override HandleAsync — the typed Executor base class wires this to ConfigureRoutes.
    public override async ValueTask<UnfinishedContentSummary> HandleAsync(
        ChatMessage message,
        IWorkflowContext context,
        CancellationToken ct = default)
    {
        // The message text is a JSON-serialized StudyPlanMessage
        var planMsg = JsonSerializer.Deserialize<StudyPlanMessage>(message.Text!)!;
        var username = planMsg.Username;
        var userId   = planMsg.UserId;

        // Use a fresh session so we don't pollute the main chat history
        var session = await _graphqlAgent.CreateSessionAsync();

        // $$""" (double-$) lets { and } appear literally; interpolations use {{expr}}
        var prompt = $$"""
            You are gathering data for a student study plan. Execute these steps in order using the query-graphql tool:

            1. Get course enrollments for userId: "{{userId}}"
            2. Get all course content already finished by student username: "{{username}}"
            3. For each enrollment where CompletionPercent < 100, get the content IDs that are NOT finished for that SubjectID and LearningLevelID
            4. Get the content details (titles) for those unfinished content IDs

            After completing all queries, return ONLY a JSON object — no extra explanation — in this exact format:
            {
              "hasUnfinishedContent": true,
              "unfinishedCourses": [
                {
                  "subjectId": "...",
                  "levelName": "...",
                  "completionPercent": 75.0,
                  "contents": [
                    { "contentId": "...", "title": "..." }
                  ]
                }
              ]
            }

            If the student has no unfinished content, return:
            { "hasUnfinishedContent": false, "unfinishedCourses": [] }
            """;

        var response = await _graphqlAgent.RunAsync(prompt, session);
        var text = response.Text.Trim();

        // Extract JSON block from the agent's response
        var json = ExtractJson(text);
        var hasContent = true;

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.TryGetProperty("hasUnfinishedContent", out var prop))
                hasContent = prop.GetBoolean();
        }
        catch
        {
            // If JSON parsing fails, treat the raw text as the data and proceed
            json = text;
        }

        return new UnfinishedContentSummary(username, hasContent, json);
    }

    /// <summary>Extracts the first complete JSON object from a string.</summary>
    private static string ExtractJson(string text)
    {
        var start = text.IndexOf('{');
        var end   = text.LastIndexOf('}');
        return start >= 0 && end > start ? text[start..(end + 1)] : text;
    }
}

// ---------------------------------------------------------------------------
// Executor 2a: Creates the study plan (reached when HasContent = true)
//
// Inherits Executor<TInput> — typed base class for void-returning handlers.
// ---------------------------------------------------------------------------

/// <summary>
/// Receives the structured unfinished content summary, sends it to the PlannerAgent
/// (pure reasoning, no MCP tools), and yields the study plan as workflow output.
/// </summary>
internal sealed class PlannerAgentExecutor : Executor<UnfinishedContentSummary>
{
    private readonly AIAgent _plannerAgent;

    public PlannerAgentExecutor(AIAgent plannerAgent) : base("PlannerAgent")
        => _plannerAgent = plannerAgent;

    public override async ValueTask HandleAsync(
        UnfinishedContentSummary summary,
        IWorkflowContext context,
        CancellationToken ct = default)
    {
        var prompt = $"""
            Here is the unfinished course data for student "{summary.Username}" in JSON format:

            {summary.DataJson}

            Based on this data, create a structured weekly study plan. The plan should:
            - Group related content by subject and learning level
            - Prioritize subjects with lower completion percentages first
            - Suggest a realistic number of content items per day (2-3 items max)
            - Use clear headings (Week 1, Day 1: ..., etc.)
            - Be encouraging and supportive in tone

            Output the study plan as plain readable text.
            """;

        var response = await _plannerAgent.RunAsync(prompt);
        await context.YieldOutputAsync(response.Text);
    }
}

// ---------------------------------------------------------------------------
// Executor 2b: All content is complete (reached when HasContent = false)
// ---------------------------------------------------------------------------

/// <summary>
/// Short-circuits the workflow when the student has no unfinished content.
/// </summary>
internal sealed class AllDoneExecutor : Executor<UnfinishedContentSummary>
{
    public AllDoneExecutor() : base("AllDone") { }

    public override async ValueTask HandleAsync(
        UnfinishedContentSummary summary,
        IWorkflowContext context,
        CancellationToken ct = default)
    {
        await context.YieldOutputAsync(
            $"Great news, {summary.Username}! " +
            "You have completed all enrolled course content. " +
            "Consider exploring new subjects to continue your learning journey!");
    }
}

// ---------------------------------------------------------------------------
// Workflow runner — builds and executes the full graph
// ---------------------------------------------------------------------------

public static class StudyPlanWorkflowRunner
{
    /// <summary>
    /// Builds the workflow graph and runs it for the given student.
    /// Returns the study plan text (or an "all done" message).
    /// </summary>
    public static async Task<string> RunAsync(
        string username,
        string userId,
        AIAgent graphqlAgent,
        AIAgent plannerAgent)
    {
        // Instantiate executors — each wraps one AI agent or pure logic
        var dataGathering = new DataGatheringExecutor(graphqlAgent);
        var planner       = new PlannerAgentExecutor(plannerAgent);
        var allDone       = new AllDoneExecutor();

        // Build the directed workflow graph:
        //
        //   [ChatMessage] ──▶ DataGatheringExecutor
        //                             │
        //          ┌─────────────────┤
        //          │ HasContent=true  ──▶ PlannerAgentExecutor ──▶ output
        //          │ HasContent=false ──▶ AllDoneExecutor       ──▶ output
        //
        var workflow = new WorkflowBuilder(dataGathering)
            .AddEdge<UnfinishedContentSummary>(dataGathering, planner,
                condition: s => s != null && s.HasContent)
            .AddEdge<UnfinishedContentSummary>(dataGathering, allDone,
                condition: s => s != null && !s.HasContent)
            .WithOutputFrom(planner, allDone)
            .Build();

        // Serialize the student info into the initial ChatMessage (workflow entry point)
        var planMsg      = JsonSerializer.Serialize(new StudyPlanMessage(username, userId));
        var inputMessage = new ChatMessage(ChatRole.User, planMsg);

        // Execute and collect output
        var result = await InProcessExecution.RunAsync(workflow, inputMessage);

        var sb = new StringBuilder();
        foreach (var evt in result.NewEvents)
        {
            if (evt is WorkflowOutputEvent output)
                sb.AppendLine(output.Data?.ToString());
        }

        var plan = sb.ToString().Trim();
        return string.IsNullOrWhiteSpace(plan)
            ? "Unable to generate a study plan at this time. Please try again."
            : plan;
    }
}
