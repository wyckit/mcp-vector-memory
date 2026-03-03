namespace McpVectorMemory.Services;

public interface IEmbeddingService
{
    float[] Embed(string text);
    int Dimensions { get; }
}
