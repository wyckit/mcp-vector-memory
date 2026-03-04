using System.ComponentModel;
using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using ModelContextProtocol.Server;

namespace McpVectorMemory.Tools;

/// <summary>
/// MCP tools for core memory operations: store, search, delete (enhanced).
/// </summary>
[McpServerToolType]
public sealed class CoreMemoryTools
{
    private readonly CognitiveIndex _index;
    private readonly PhysicsEngine _physics;
    private readonly IEmbeddingService _embedding;

    public CoreMemoryTools(CognitiveIndex index, PhysicsEngine physics, IEmbeddingService embedding)
    {
        _index = index;
        _physics = physics;
        _embedding = embedding;
    }

    [McpServerTool(Name = "store_memory")]
    [Description("Store a cognitive memory entry with namespace isolation, categorical metadata, and lifecycle tracking. Entry starts in STM by default.")]
    public string StoreMemory(
        [Description("Unique identifier for this memory entry.")] string id,
        [Description("Namespace (e.g. 'work', 'personal').")] string ns,
        [Description("The original text the vector was derived from.")] string? text = null,
        [Description("The float vector embedding as an array of numbers.")] float[]? vector = null,
        [Description("Category within namespace (e.g. 'meeting-notes').")] string? category = null,
        [Description("Optional metadata as a JSON object with string keys and values.")] Dictionary<string, string>? metadata = null,
        [Description("Initial lifecycle state: 'stm' (default), 'ltm', or 'archived'.")] string? lifecycleState = null)
    {
        try
        {
            var resolved = ResolveVector(vector, text);
            var entry = new CognitiveEntry(id, resolved, ns, text, category, metadata,
                lifecycleState ?? "stm");
            _index.Upsert(entry);
            return $"Stored entry '{id}' ({resolved.Length}-dim vector) in namespace '{ns}'.";
        }
        catch (ArgumentException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [McpServerTool(Name = "search_memory")]
    [Description("Namespace-scoped vector similarity search with lifecycle awareness, summary-first mode, and optional physics-based gravity re-ranking (slingshot output).")]
    public object SearchMemory(
        [Description("Namespace to search.")] string ns,
        [Description("The original text to search for.")] string? text = null,
        [Description("The query vector embedding as an array of numbers.")] float[]? vector = null,
        [Description("Maximum number of results to return (default: 5).")] int k = 5,
        [Description("Minimum cosine-similarity score threshold (default: 0).")] float minScore = 0f,
        [Description("Filter by category within namespace.")] string? category = null,
        [Description("Comma-separated lifecycle states to include (default: 'stm,ltm').")] string? includeStates = null,
        [Description("Prioritize cluster summaries in results (default: false).")] bool summaryFirst = false,
        [Description("When true, return physics-based slingshot output (Asteroid + Sun) instead of flat list.")] bool usePhysics = false)
    {
        float[] resolved;
        try
        {
            resolved = ResolveVector(vector, text);
        }
        catch (ArgumentException ex)
        {
            return $"Error: {ex.Message}";
        }

        var states = includeStates is not null
            ? new HashSet<string>(includeStates.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            : new HashSet<string> { "stm", "ltm" };

        var results = _index.Search(resolved, ns, k, minScore, category, states, summaryFirst);

        // Side effect: record access for returned entries
        foreach (var result in results)
            _index.RecordAccess(result.Id);

        if (usePhysics && results.Count > 0)
            return _physics.Slingshot(results);

        return results;
    }

    [McpServerTool(Name = "delete_memory")]
    [Description("Delete a memory entry by ID, with cascading removal of graph edges and cluster memberships.")]
    public string DeleteMemory(
        [Description("The identifier of the entry to delete.")] string id,
        KnowledgeGraph graph,
        ClusterManager clusters)
    {
        // Cascade: remove graph edges (safe even if entry doesn't exist)
        int edgesRemoved = graph.RemoveAllEdgesForEntry(id);

        // Cascade: remove from clusters
        clusters.RemoveEntryFromAllClusters(id);

        // Remove the entry itself — check return value to avoid TOCTOU
        if (!_index.Delete(id))
            return $"Entry '{id}' not found.";

        return $"Deleted entry '{id}'. Removed {edgesRemoved} edge(s) and cleaned cluster memberships.";
    }

    private float[] ResolveVector(float[]? vector, string? text)
    {
        if (vector is not null && vector.Length > 0)
            return vector;

        if (!string.IsNullOrWhiteSpace(text))
            return _embedding.Embed(text);

        throw new ArgumentException("Either 'vector' or 'text' must be provided.");
    }
}
