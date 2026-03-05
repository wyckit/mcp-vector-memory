using SmartComponents.LocalEmbeddings;

namespace McpVectorMemory.Core.Services;

public sealed class LocalEmbeddingService : IEmbeddingService, IDisposable
{
    private readonly LocalEmbedder _embedder = new();
    private readonly SemaphoreSlim _lock = new(1, 1);

    public int Dimensions => 384;

    public float[] Embed(string text)
    {
        _lock.Wait();
        try
        {
            var embedding = _embedder.Embed(text);
            return embedding.Values.ToArray();
        }
        finally { _lock.Release(); }
    }

    public void Dispose()
    {
        _lock.Dispose();
        _embedder.Dispose();
    }
}
