using System.Text.Json.Serialization;

namespace McpVectorMemory.Core.Models;

/// <summary>
/// Serialization container for all data within a single namespace.
/// Edges and clusters are stored in separate global files (_edges.json, _clusters.json).
/// </summary>
public sealed class NamespaceData
{
    [JsonPropertyName("entries")]
    public List<CognitiveEntry> Entries { get; set; } = new();
}
