using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace McpVectorMemory.Services;

public sealed class EmbeddingWarmupService : IHostedService
{
    private readonly IEmbeddingService _embedding;
    private readonly ILogger<EmbeddingWarmupService> _logger;

    public EmbeddingWarmupService(IEmbeddingService embedding, ILogger<EmbeddingWarmupService> logger)
    {
        _embedding = embedding;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Warming up embedding model ({Dimensions}-dim)...", _embedding.Dimensions);
        _embedding.Embed("warmup");
        _logger.LogInformation("Embedding model ready.");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
