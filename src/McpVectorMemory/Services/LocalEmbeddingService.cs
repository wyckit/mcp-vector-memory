using SmartComponents.LocalEmbeddings;

namespace McpVectorMemory.Services;

public sealed class LocalEmbeddingService : IEmbeddingService, IDisposable
{
    private readonly LocalEmbedder _embedder = new();

    public int Dimensions => 384;

    public float[] Embed(string text)
    {
        var embedding = _embedder.Embed(text);
        return embedding.Values.ToArray();
    }

    public void Dispose() => _embedder.Dispose();
}
