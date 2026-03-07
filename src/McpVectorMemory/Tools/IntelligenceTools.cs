using System.ComponentModel;
using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using ModelContextProtocol.Server;

namespace McpVectorMemory.Tools;

/// <summary>
/// MCP tools for intelligence features: duplicate detection, contradiction surfacing, reversible collapse.
/// </summary>
[McpServerToolType]
public sealed class IntelligenceTools
{
    private readonly CognitiveIndex _index;
    private readonly KnowledgeGraph _graph;
    private readonly IEmbeddingService _embedding;
    private readonly AccretionScanner _scanner;
    private readonly ClusterManager _clusters;
    private readonly LifecycleEngine _lifecycle;

    public IntelligenceTools(
        CognitiveIndex index, KnowledgeGraph graph, IEmbeddingService embedding,
        AccretionScanner scanner, ClusterManager clusters, LifecycleEngine lifecycle)
    {
        _index = index;
        _graph = graph;
        _embedding = embedding;
        _scanner = scanner;
        _clusters = clusters;
        _lifecycle = lifecycle;
    }

    [McpServerTool(Name = "detect_duplicates")]
    [Description("Find near-duplicate memory entries within a namespace by pairwise cosine similarity. Returns pairs above the threshold sorted by similarity.")]
    public object DetectDuplicates(
        [Description("Namespace to scan.")] string ns,
        [Description("Cosine similarity threshold (default: 0.95). Entries above this are flagged as duplicates.")] float threshold = 0.95f,
        [Description("Filter by category.")] string? category = null,
        [Description("Comma-separated lifecycle states to include (default: 'stm,ltm').")] string? includeStates = null)
    {
        if (threshold < 0f || threshold > 1f)
            return "Error: Threshold must be between 0 and 1.";

        var states = includeStates is not null
            ? new HashSet<string>(includeStates.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            : new HashSet<string> { "stm", "ltm" };

        var raw = _index.FindDuplicates(ns, threshold, category, states);

        var pairs = new List<DuplicatePair>(raw.Count);
        foreach (var (idA, idB, sim) in raw)
        {
            var a = _index.Get(idA, ns);
            var b = _index.Get(idB, ns);
            if (a is null || b is null) continue;

            pairs.Add(new DuplicatePair(
                new CognitiveEntryInfo(a.Id, a.Text, a.Ns, a.Category, a.LifecycleState),
                new CognitiveEntryInfo(b.Id, b.Text, b.Ns, b.Category, b.LifecycleState),
                sim));
        }

        var scannedCount = _index.CountInNamespace(ns);
        return new DuplicateDetectionResult(scannedCount, pairs, threshold);
    }

    [McpServerTool(Name = "find_contradictions")]
    [Description("Surface contradictions in a namespace: entries explicitly linked with 'contradicts' edges, plus high-similarity entry pairs that may need review. Set a query to find contradictions relevant to a topic.")]
    public object FindContradictions(
        [Description("Namespace to search.")] string ns,
        [Description("Optional topic text to focus contradiction search.")] string? topic = null,
        [Description("Cosine similarity threshold for potential contradiction detection (default: 0.8).")] float similarityThreshold = 0.8f)
    {
        // Part 1: Get explicit contradiction edges from the knowledge graph
        var graphContradictions = _graph.GetContradictions(ns);
        var contradictions = new List<ContradictionInfo>();

        foreach (var (edge, source, target) in graphContradictions)
        {
            if (source is null || target is null) continue;

            // Compute similarity between the two entries
            float sim = 0f;
            if (source.Vector.Length == target.Vector.Length)
            {
                float sourceNorm = CognitiveIndex.Norm(source.Vector);
                float targetNorm = CognitiveIndex.Norm(target.Vector);
                if (sourceNorm > 0f && targetNorm > 0f)
                    sim = CognitiveIndex.Dot(source.Vector, target.Vector) / (sourceNorm * targetNorm);
            }

            contradictions.Add(new ContradictionInfo(
                new CognitiveEntryInfo(source.Id, source.Text, source.Ns, source.Category, source.LifecycleState),
                new CognitiveEntryInfo(target.Id, target.Text, target.Ns, target.Category, target.LifecycleState),
                sim, "graph_edge"));
        }
        int graphCount = contradictions.Count;

        // Part 2: If a topic is provided, find high-similarity entries that might contradict
        int highSimCount = 0;
        if (topic is not null)
        {
            var vector = _embedding.Embed(topic);
            var results = _index.Search(vector, ns, k: 20, minScore: similarityThreshold);

            // Check for pairs among the results that are very similar to each other
            // but aren't already linked — these are candidates for contradiction review
            for (int i = 0; i < results.Count; i++)
            {
                for (int j = i + 1; j < results.Count; j++)
                {
                    var a = _index.Get(results[i].Id, ns);
                    var b = _index.Get(results[j].Id, ns);
                    if (a is null || b is null) continue;
                    if (a.Vector.Length != b.Vector.Length) continue;

                    float aNorm = CognitiveIndex.Norm(a.Vector);
                    float bNorm = CognitiveIndex.Norm(b.Vector);
                    if (aNorm == 0f || bNorm == 0f) continue;

                    float pairSim = CognitiveIndex.Dot(a.Vector, b.Vector) / (aNorm * bNorm);
                    if (pairSim < similarityThreshold) continue;

                    // Skip if this pair is already in the graph contradictions
                    bool alreadyKnown = contradictions.Any(c =>
                        (c.EntryA.Id == a.Id && c.EntryB.Id == b.Id) ||
                        (c.EntryA.Id == b.Id && c.EntryB.Id == a.Id));
                    if (alreadyKnown) continue;

                    contradictions.Add(new ContradictionInfo(
                        new CognitiveEntryInfo(a.Id, a.Text, a.Ns, a.Category, a.LifecycleState),
                        new CognitiveEntryInfo(b.Id, b.Text, b.Ns, b.Category, b.LifecycleState),
                        pairSim, "high_similarity"));
                    highSimCount++;
                }
            }
        }

        return new ContradictionResult(contradictions, graphCount, highSimCount);
    }

    [McpServerTool(Name = "uncollapse_cluster")]
    [Description("Reverse a previously executed accretion collapse: restore archived members to their pre-collapse lifecycle state, delete the summary entry, and clean up the cluster.")]
    public string UncollapseCluster(
        [Description("The collapse ID to reverse.")] string collapseId)
    {
        return _scanner.UndoCollapse(collapseId, _lifecycle, _clusters);
    }

    [McpServerTool(Name = "list_collapse_history")]
    [Description("List all reversible collapse records for a namespace.")]
    public IReadOnlyList<CollapseRecord> ListCollapseHistory(
        [Description("Namespace to list collapse history for.")] string ns)
    {
        return _scanner.GetCollapseHistory(ns);
    }
}
