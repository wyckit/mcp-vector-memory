using System.ComponentModel;
using McpEngramMemory.Core.Models;
using McpEngramMemory.Core.Services;
using McpEngramMemory.Core.Services.Evaluation;
using McpEngramMemory.Core.Services.Experts;
using ModelContextProtocol.Server;

namespace McpEngramMemory.Tools;

/// <summary>
/// MCP tools for Expert Instantiation and Semantic Routing.
/// dispatch_task routes queries to specialized expert namespaces via semantic similarity.
/// create_expert registers new expert personas in the meta-index.
/// get_domain_tree shows the hierarchical expert domain tree.
/// </summary>
[McpServerToolType]
public sealed class ExpertTools
{
    private readonly ExpertDispatcher _dispatcher;
    private readonly CognitiveIndex _index;
    private readonly IEmbeddingService _embedding;
    private readonly MetricsCollector _metrics;

    public ExpertTools(
        ExpertDispatcher dispatcher,
        CognitiveIndex index,
        IEmbeddingService embedding,
        MetricsCollector metrics)
    {
        _dispatcher = dispatcher;
        _index = index;
        _embedding = embedding;
        _metrics = metrics;
    }

    [McpServerTool(Name = "dispatch_task")]
    [Description("Route a query to the most relevant expert namespace via semantic similarity against the meta-index. " +
        "If a qualified expert is found (cosine similarity >= threshold), returns the expert profile and top memories " +
        "from that expert's namespace as context. If no expert qualifies, returns 'needs_expert' status — " +
        "use create_expert to instantiate a specialist for the domain. " +
        "Set hierarchical=true to use coarse-to-fine tree routing through domain nodes (root → branch → leaf).")]
    public object DispatchTask(
        [Description("The core problem, question, or task to route to an expert.")] string taskDescription,
        [Description("How many memories to retrieve from the matched expert's namespace (default: 3).")] int autoSearchK = 3,
        [Description("Minimum cosine similarity threshold for a routing hit (default: 0.75). " +
            "Lower values match more broadly, higher values require stronger domain alignment.")] float threshold = 0.75f,
        [Description("When true, uses hierarchical tree routing (root → branch → leaf) instead of flat comparison. " +
            "Falls back to flat routing if no domain tree exists. Default: false.")] bool hierarchical = false)
    {
        if (string.IsNullOrWhiteSpace(taskDescription))
            return "Error: taskDescription must not be empty.";
        if (autoSearchK < 1) autoSearchK = 1;
        if (threshold <= 0f || threshold > 1f) threshold = ExpertDispatcher.DefaultThreshold;

        using var timer = _metrics.StartTimer("dispatch_task");

        var queryVector = _embedding.Embed(taskDescription);

        if (hierarchical)
        {
            var result = _dispatcher.RouteHierarchical(queryVector, topK: topK(autoSearchK), threshold: threshold);
            if (result.Status == "routed" && result.Experts.Count > 0)
                _dispatcher.RecordDispatch(result.Experts[0].ExpertId);
            return result;
        }

        var (status, experts) = _dispatcher.Route(queryVector, topK: 3, threshold: threshold);

        if (status == "needs_expert")
        {
            return new DispatchMissResult(
                "needs_expert",
                experts,
                "No qualified expert found for this domain. Use create_expert to instantiate a specialist.");
        }

        // Routed — search the best expert's namespace for context
        var bestExpert = experts[0];
        _dispatcher.RecordDispatch(bestExpert.ExpertId);

        var context = _index.Search(
            queryVector, bestExpert.TargetNamespace, k: autoSearchK);

        return new DispatchRoutedResult("routed", bestExpert, experts, context);

        // Local function to calculate topK for hierarchical routing
        static int topK(int autoSearchK) => Math.Max(autoSearchK, 3);
    }

    [McpServerTool(Name = "create_expert")]
    [Description("Instantiate a new expert namespace and register it in the semantic routing meta-index. " +
        "The persona description is embedded and used for future query routing. " +
        "Provide a detailed paragraph outlining the expert's domain, principles, and perspective. " +
        "Use level='root' or 'branch' to create domain nodes for hierarchical routing. " +
        "When parentNodeId is omitted for leaf experts, auto-classification places the expert " +
        "into the domain tree: 'auto_linked' (>=0.82 confidence), 'suggested' (0.60-0.82), " +
        "or 'unclassified' (<0.60). The placement result is included in the response.")]
    public object CreateExpert(
        [Description("Snake_case identifier for the expert (e.g., 'rust_systems_engineer', 'quantum_physicist').")] string expertId,
        [Description("Detailed paragraph describing the expert's domain expertise, specialization, and perspective. " +
            "This text is embedded and used for semantic matching during dispatch.")] string personaDescription,
        [Description("Hierarchy level: 'leaf' (default) for actual experts, 'root' for top-level domains, " +
            "'branch' for mid-level groupings. Root and branch nodes are routing-only domain nodes.")] string level = "leaf",
        [Description("Parent node ID for branch or leaf nodes to establish tree hierarchy. " +
            "Required for 'branch' level, optional for 'leaf'.")] string? parentNodeId = null)
    {
        if (string.IsNullOrWhiteSpace(expertId))
            return "Error: expertId must not be empty.";
        if (string.IsNullOrWhiteSpace(personaDescription))
            return "Error: personaDescription must not be empty.";

        using var timer = _metrics.StartTimer("create_expert");

        if (_dispatcher.ExpertExists(expertId))
            return $"Error: Expert '{expertId}' already exists. Use a different ID or update the existing expert.";

        if (level == "root" || level == "branch")
        {
            try
            {
                var result = _dispatcher.CreateDomainNode(expertId, personaDescription, level, parentNodeId);
                return new CreateExpertResult("created", result.ExpertId, result.TargetNamespace);
            }
            catch (ArgumentException ex)
            {
                return $"Error: {ex.Message}";
            }
        }

        // Leaf expert — use existing CreateExpert
        var expert = _dispatcher.CreateExpert(expertId, personaDescription);

        // If parentNodeId is provided, link explicitly (no auto-classification)
        if (parentNodeId is not null)
        {
            _dispatcher.LinkToParent(expertId, parentNodeId);
            return new CreateExpertResult("created", expert.ExpertId, expert.TargetNamespace);
        }

        // Auto-classify into domain tree when no parent specified
        var personaVector = _embedding.Embed(personaDescription);
        var placement = _dispatcher.ClassifyExpert(personaVector);

        if (placement.Status is "auto_linked" or "suggested" && placement.ParentNodeId is not null)
        {
            _dispatcher.LinkToParent(expertId, placement.ParentNodeId);
        }

        return new CreateExpertResult("created", expert.ExpertId, expert.TargetNamespace, placement);
    }

    [McpServerTool(Name = "link_to_parent")]
    [Description("Link an existing leaf expert to a parent node (root or branch) in the domain tree. " +
        "Use this to organize existing experts into the hierarchical routing topology.")]
    public object LinkToParent(
        [Description("The expert ID to link.")] string expertId,
        [Description("The parent node ID (root or branch) to link under.")] string parentNodeId)
    {
        if (string.IsNullOrWhiteSpace(expertId))
            return "Error: expertId must not be empty.";
        if (string.IsNullOrWhiteSpace(parentNodeId))
            return "Error: parentNodeId must not be empty.";

        using var timer = _metrics.StartTimer("link_to_parent");

        if (!_dispatcher.LinkToParent(expertId, parentNodeId))
            return $"Error: Expert '{expertId}' or parent '{parentNodeId}' not found.";

        return new { status = "linked", expertId, parentNodeId };
    }

    [McpServerTool(Name = "get_domain_tree")]
    [Description("Get the full expert domain tree showing root domains, branches, and leaf experts " +
        "with their hierarchical relationships. Useful for understanding the routing topology.")]
    public object GetDomainTree()
    {
        using var timer = _metrics.StartTimer("get_domain_tree");
        return _dispatcher.GetDomainTree();
    }
}
