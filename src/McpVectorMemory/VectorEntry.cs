using System.Collections.ObjectModel;
using System.Text.Json.Serialization;

namespace McpVectorMemory;

/// <summary>
/// Represents a stored vector with an identifier and optional metadata.
/// </summary>
public sealed class VectorEntry
{
    [JsonPropertyName("id")]
    public string Id { get; }

    [JsonIgnore]
    public float[] Vector { get; }

    [JsonPropertyName("text")]
    public string? Text { get; }

    [JsonPropertyName("metadata")]
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>UTC timestamp when this entry was created (or last upserted).</summary>
    [JsonIgnore]
    public DateTime CreatedAtUtc { get; }

    public VectorEntry(string id, float[] vector, string? text = null,
        Dictionary<string, string>? metadata = null, DateTime? createdAtUtc = null)
    {
        if (string.IsNullOrWhiteSpace(id))
            throw new ArgumentException("Id must not be empty.", nameof(id));
        if (vector is null || vector.Length == 0)
            throw new ArgumentException("Vector must not be null or empty.", nameof(vector));

        // Clone first so the magnitude check operates on our own copy
        Id = id;
        Vector = (float[])vector.Clone();

        if (VectorMath.Norm(Vector) == 0f)
            throw new ArgumentException("Vector must not be zero-magnitude.", nameof(vector));
        Text = text;
        Metadata = new ReadOnlyDictionary<string, string>(
            metadata is not null ? new Dictionary<string, string>(metadata) : new Dictionary<string, string>());
        CreatedAtUtc = createdAtUtc ?? DateTime.UtcNow;
    }
}
