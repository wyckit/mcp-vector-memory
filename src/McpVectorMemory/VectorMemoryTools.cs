using System.ComponentModel;
using ModelContextProtocol.Server;

namespace McpVectorMemory;

/// <summary>
/// MCP tool class exposing vector-memory operations to an LLM.
/// </summary>
[McpServerToolType]
public sealed class VectorMemoryTools
{
    private readonly VectorIndex _index;

    public VectorMemoryTools(VectorIndex index)
    {
        ArgumentNullException.ThrowIfNull(index);
        _index = index;
    }

    /// <summary>
    /// Stores (or replaces) a vector in the memory index together with its
    /// source text and optional metadata key-value pairs.
    /// </summary>
    [McpServerTool(Name = "store_memory")]
    [Description("Store a vector embedding together with its text and optional metadata.")]
    public string StoreMemory(
        [Description("Unique identifier for this memory entry.")] string id,
        [Description("The float vector embedding as an array of numbers.")] float[] vector,
        [Description("The original text the vector was derived from.")] string? text = null,
        [Description("Optional metadata as a JSON object with string keys and values.")] Dictionary<string, string>? metadata = null)
    {
        try
        {
            var entry = new VectorEntry(id, vector, text, metadata);
            _index.Upsert(entry);
            return $"Stored entry '{id}' ({vector.Length}-dim vector).";
        }
        catch (ArgumentException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Stores multiple vector entries in a single batch operation.
    /// More efficient than calling store_memory in a loop.
    /// </summary>
    [McpServerTool(Name = "store_memories")]
    [Description("Store multiple vector embeddings in a single batch operation. Each item needs an id, vector, and optionally text and metadata.")]
    public string StoreMemories(
        [Description("Array of memory entries to store. Each must have 'id' (string) and 'vector' (float[]), and optionally 'text' (string) and 'metadata' (object).")] MemoryInput[] entries)
    {
        if (entries is null || entries.Length == 0)
            return "Error: entries must not be null or empty.";

        try
        {
            var vectorEntries = new List<VectorEntry>(entries.Length);
            foreach (var e in entries)
                vectorEntries.Add(new VectorEntry(e.Id, e.Vector, e.Text, e.Metadata));

            int count = _index.BulkUpsert(vectorEntries);
            return $"Stored {count} entries.";
        }
        catch (ArgumentException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Searches for the nearest neighbors of a query vector.
    /// </summary>
    [McpServerTool(Name = "search_memory")]
    [Description("Find the most similar stored memories for a query vector.")]
    public object SearchMemory(
        [Description("The query vector embedding as an array of numbers.")] float[] vector,
        [Description("Maximum number of results to return (default: 5).")] int k = 5,
        [Description("Minimum cosine-similarity score threshold between -1 and 1 (default: 0).")] float minScore = 0f,
        [Description("Number of top results to skip for pagination (default: 0).")] int offset = 0)
    {
        try
        {
            return _index.Search(vector, k, minScore, offset);
        }
        catch (ArgumentException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Deletes a stored memory entry by its identifier.
    /// </summary>
    [McpServerTool(Name = "delete_memory")]
    [Description("Delete a stored memory entry by its unique identifier.")]
    public string DeleteMemory(
        [Description("The identifier of the entry to delete.")] string id)
    {
        bool removed = _index.Delete(id);
        return removed
            ? $"Deleted entry '{id}'."
            : $"Entry '{id}' not found.";
    }

    /// <summary>
    /// Deletes multiple stored memory entries by their identifiers in a single batch.
    /// </summary>
    [McpServerTool(Name = "delete_memories")]
    [Description("Delete multiple stored memory entries by their identifiers in a single batch operation.")]
    public string DeleteMemories(
        [Description("Array of entry identifiers to delete.")] string[] ids)
    {
        if (ids is null || ids.Length == 0)
            return "Error: ids must not be null or empty.";

        int deleted = _index.BulkDelete(ids);
        return $"Deleted {deleted} of {ids.Length} entries.";
    }
}

/// <summary>
/// Input DTO for the store_memories bulk tool.
/// </summary>
public sealed class MemoryInput
{
    public string Id { get; set; } = "";
    public float[] Vector { get; set; } = Array.Empty<float>();
    public string? Text { get; set; }
    public Dictionary<string, string>? Metadata { get; set; }
}
