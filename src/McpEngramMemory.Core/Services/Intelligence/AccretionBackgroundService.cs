using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace McpEngramMemory.Core.Services.Intelligence;

/// <summary>
/// Background service that periodically scans all namespaces for dense LTM clusters
/// using DBSCAN and creates pending collapses for LLM-driven summarization.
/// </summary>
public sealed class AccretionBackgroundService : BackgroundService
{
    private readonly AccretionScanner _scanner;
    private readonly CognitiveIndex _index;
    private readonly ClusterManager _clusters;
    private readonly IEmbeddingService _embedding;
    private readonly ILogger<AccretionBackgroundService> _logger;

    /// <summary>Default interval between accretion scans.</summary>
    public static readonly TimeSpan DefaultInterval = TimeSpan.FromMinutes(30);

    /// <summary>Configurable interval (for testing).</summary>
    public TimeSpan Interval { get; set; } = DefaultInterval;

    public AccretionBackgroundService(
        AccretionScanner scanner, CognitiveIndex index, ClusterManager clusters,
        IEmbeddingService embedding, ILogger<AccretionBackgroundService> logger)
    {
        _scanner = scanner;
        _index = index;
        _clusters = clusters;
        _embedding = embedding;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Accretion background service started (interval: {Interval})", Interval);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(Interval, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            try
            {
                var namespaces = _index.GetNamespaces();
                int totalClusters = 0;

                foreach (var ns in namespaces)
                {
                    var result = _scanner.ScanNamespace(ns,
                        autoSummarize: true, clusters: _clusters, embedding: _embedding);
                    totalClusters += result.ClustersDetected;

                    if (result.NewCollapses.Count > 0)
                    {
                        _logger.LogInformation(
                            "Accretion scan: namespace '{Ns}' detected {Clusters} new collapse(s)",
                            ns, result.NewCollapses.Count);
                    }

                    if (result.AutoSummaries is { Count: > 0 })
                    {
                        _logger.LogInformation(
                            "Auto-summarized {Count} cluster(s) in namespace '{Ns}'",
                            result.AutoSummaries.Count, ns);
                    }
                }

                _logger.LogDebug("Accretion scan completed across {Count} namespace(s), {Clusters} total clusters",
                    namespaces.Count, totalClusters);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during accretion scan");
            }
        }

        _logger.LogInformation("Accretion background service stopped");
    }
}
