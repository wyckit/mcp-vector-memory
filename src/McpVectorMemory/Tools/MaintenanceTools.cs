using System.ComponentModel;
using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using ModelContextProtocol.Server;

namespace McpVectorMemory.Tools;

/// <summary>
/// MCP tools for maintenance operations: rebuild embeddings.
/// </summary>
[McpServerToolType]
public sealed class MaintenanceTools
{
    private readonly CognitiveIndex _index;
    private readonly IEmbeddingService _embedding;
    private readonly MetricsCollector _metrics;

    public MaintenanceTools(CognitiveIndex index, IEmbeddingService embedding, MetricsCollector metrics)
    {
        _index = index;
        _embedding = embedding;
        _metrics = metrics;
    }

    [McpServerTool(Name = "rebuild_embeddings")]
    [Description("Re-embed all entries in one or all namespaces using the current embedding model. " +
        "Use after upgrading the embedding model to regenerate vectors from stored text. " +
        "Entries without text are skipped. Preserves all metadata, lifecycle state, and timestamps.")]
    public object RebuildEmbeddings(
        [Description("Namespace to rebuild ('*' for all namespaces, default: '*').")] string ns = "*")
    {
        using var timer = _metrics.StartTimer("rebuild_embeddings");

        var namespaces = ns == "*"
            ? _index.GetNamespaces()
            : new[] { ns };

        var results = new List<RebuildNamespaceResult>();
        int totalUpdated = 0, totalSkipped = 0;

        foreach (var namespaceName in namespaces)
        {
            var (updated, skipped) = _index.RebuildEmbeddings(namespaceName, _embedding);
            results.Add(new RebuildNamespaceResult(namespaceName, updated, skipped));
            totalUpdated += updated;
            totalSkipped += skipped;
        }

        return new RebuildEmbeddingsResult(
            totalUpdated, totalSkipped, results.Count, results, _embedding.Dimensions);
    }
}
