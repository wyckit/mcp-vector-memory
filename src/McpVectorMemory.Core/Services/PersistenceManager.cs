using System.Text.Json;
using McpVectorMemory.Core.Models;
using Microsoft.Extensions.Logging;

namespace McpVectorMemory.Core.Services;

/// <summary>
/// JSON file-based persistence per namespace with debounced async writes.
/// </summary>
public sealed class PersistenceManager : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private readonly string _basePath;
    private readonly object _timerLock = new();
    private readonly TimeSpan _debounceDelay;
    private readonly ILogger<PersistenceManager>? _logger;
    private bool _disposed;

    // Pending namespace saves (keyed by namespace name)
    private readonly Dictionary<string, (Timer Timer, Func<NamespaceData> DataProvider)> _pendingNsSaves = new();

    // Pending global edge save (separate from namespace saves to avoid dummy-data overwrite)
    private Timer? _pendingEdgeTimer;
    private Func<List<GraphEdge>>? _pendingEdgeProvider;

    // Pending cluster save
    private Timer? _pendingClusterTimer;
    private Func<List<SemanticCluster>>? _pendingClusterProvider;

    public PersistenceManager(string? basePath = null, int debounceMs = 500, ILogger<PersistenceManager>? logger = null)
    {
        _basePath = basePath ?? Path.Combine(AppContext.BaseDirectory, "data");
        _debounceDelay = TimeSpan.FromMilliseconds(debounceMs);
        _logger = logger;
        Directory.CreateDirectory(_basePath);
    }

    /// <summary>
    /// Load namespace data from disk. Returns empty data if file does not exist or is corrupted.
    /// </summary>
    public NamespaceData LoadNamespace(string ns)
    {
        var path = GetNamespacePath(ns);
        if (!File.Exists(path))
            return new NamespaceData();

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<NamespaceData>(json, JsonOptions) ?? new NamespaceData();
        }
        catch (JsonException ex)
        {
            _logger?.LogWarning(ex, "Corrupted JSON in namespace '{Namespace}', returning empty data", ns);
            return new NamespaceData();
        }
    }

    /// <summary>
    /// Load global (cross-namespace) edges from disk. Returns empty list if file is corrupted.
    /// </summary>
    public List<GraphEdge> LoadGlobalEdges()
    {
        var path = Path.Combine(_basePath, "_edges.json");
        if (!File.Exists(path))
            return new();

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<List<GraphEdge>>(json, JsonOptions) ?? new();
        }
        catch (JsonException ex)
        {
            _logger?.LogWarning(ex, "Corrupted JSON in edges file, returning empty data");
            return new();
        }
    }

    /// <summary>
    /// Load clusters from disk. Returns empty list if file is corrupted.
    /// </summary>
    public List<SemanticCluster> LoadClusters()
    {
        var path = Path.Combine(_basePath, "_clusters.json");
        if (!File.Exists(path))
            return new();

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<List<SemanticCluster>>(json, JsonOptions) ?? new();
        }
        catch (JsonException ex)
        {
            _logger?.LogWarning(ex, "Corrupted JSON in clusters file, returning empty data");
            return new();
        }
    }

    /// <summary>
    /// Schedule a debounced save of namespace data.
    /// </summary>
    public void ScheduleSave(string ns, Func<NamespaceData> dataProvider)
    {
        lock (_timerLock)
        {
            if (_disposed) return;

            if (_pendingNsSaves.TryGetValue(ns, out var existing))
                existing.Timer.Dispose();

            var timer = new Timer(_ =>
            {
                Func<NamespaceData>? provider = null;
                lock (_timerLock)
                {
                    if (_pendingNsSaves.TryGetValue(ns, out var entry))
                    {
                        provider = entry.DataProvider;
                        entry.Timer.Dispose();
                        _pendingNsSaves.Remove(ns);
                    }
                }
                if (provider is not null)
                    WriteNamespace(ns, provider);
            }, null, _debounceDelay, Timeout.InfiniteTimeSpan);

            _pendingNsSaves[ns] = (timer, dataProvider);
        }
    }

    /// <summary>
    /// Schedule a debounced save of global edges.
    /// Data provider should return a pre-captured snapshot (no lock re-entry).
    /// </summary>
    public void ScheduleSaveGlobalEdges(Func<List<GraphEdge>> dataProvider)
    {
        lock (_timerLock)
        {
            if (_disposed) return;

            _pendingEdgeTimer?.Dispose();
            _pendingEdgeProvider = dataProvider;

            _pendingEdgeTimer = new Timer(_ =>
            {
                Func<List<GraphEdge>>? provider;
                lock (_timerLock)
                {
                    provider = _pendingEdgeProvider;
                    _pendingEdgeProvider = null;
                    _pendingEdgeTimer?.Dispose();
                    _pendingEdgeTimer = null;
                }
                if (provider is not null)
                    WriteGlobalEdges(provider);
            }, null, _debounceDelay, Timeout.InfiniteTimeSpan);
        }
    }

    /// <summary>
    /// Schedule a debounced save of clusters.
    /// Data provider should return a pre-captured snapshot (no lock re-entry).
    /// </summary>
    public void ScheduleSaveClusters(Func<List<SemanticCluster>> dataProvider)
    {
        lock (_timerLock)
        {
            if (_disposed) return;

            _pendingClusterTimer?.Dispose();
            _pendingClusterProvider = dataProvider;

            _pendingClusterTimer = new Timer(_ =>
            {
                Func<List<SemanticCluster>>? provider;
                lock (_timerLock)
                {
                    provider = _pendingClusterProvider;
                    _pendingClusterProvider = null;
                    _pendingClusterTimer?.Dispose();
                    _pendingClusterTimer = null;
                }
                if (provider is not null)
                    WriteClusters(provider);
            }, null, _debounceDelay, Timeout.InfiniteTimeSpan);
        }
    }

    /// <summary>
    /// Synchronously save namespace data.
    /// </summary>
    public void SaveNamespaceSync(string ns, NamespaceData data)
    {
        var path = GetNamespacePath(ns);
        var json = JsonSerializer.Serialize(data, JsonOptions);
        AtomicWriteAllText(path, json);
    }

    /// <summary>
    /// Get all namespace names from existing files on disk.
    /// </summary>
    public IReadOnlyList<string> GetPersistedNamespaces()
    {
        if (!Directory.Exists(_basePath))
            return Array.Empty<string>();

        return Directory.GetFiles(_basePath, "*.json")
            .Select(Path.GetFileNameWithoutExtension)
            .Where(n => n != null && n != "_edges" && n != "_clusters")
            .Select(n => n!)
            .ToList();
    }

    /// <summary>
    /// Flush all pending saves immediately and synchronously.
    /// </summary>
    public void Flush()
    {
        List<(string Ns, Func<NamespaceData> Provider)> pendingNs;
        Func<List<GraphEdge>>? edgeProvider;
        Func<List<SemanticCluster>>? clusterProvider;

        lock (_timerLock)
        {
            pendingNs = _pendingNsSaves
                .Select(kv => (kv.Key, kv.Value.DataProvider))
                .ToList();
            foreach (var (_, (timer, _)) in _pendingNsSaves)
                timer.Dispose();
            _pendingNsSaves.Clear();

            edgeProvider = _pendingEdgeProvider;
            _pendingEdgeProvider = null;
            _pendingEdgeTimer?.Dispose();
            _pendingEdgeTimer = null;

            clusterProvider = _pendingClusterProvider;
            _pendingClusterProvider = null;
            _pendingClusterTimer?.Dispose();
            _pendingClusterTimer = null;
        }

        foreach (var (ns, provider) in pendingNs)
            WriteNamespace(ns, provider);

        if (edgeProvider is not null)
            WriteGlobalEdges(edgeProvider);

        if (clusterProvider is not null)
            WriteClusters(clusterProvider);
    }

    public void Dispose()
    {
        lock (_timerLock)
        {
            if (_disposed) return;
            _disposed = true;
        }
        Flush();
    }

    private string GetNamespacePath(string ns)
    {
        // Sanitize namespace for filename safety
        var safe = string.Join("_", ns.Split(Path.GetInvalidFileNameChars()));
        safe = safe.Replace("..", "_"); // Prevent path traversal
        var path = Path.Combine(_basePath, $"{safe}.json");

        // Guard against path traversal
        if (!Path.GetFullPath(path).StartsWith(Path.GetFullPath(_basePath), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"Invalid namespace: '{ns}'");

        return path;
    }

    private void WriteNamespace(string ns, Func<NamespaceData> provider)
    {
        try
        {
            var data = provider();
            SaveNamespaceSync(ns, data);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to save namespace '{Namespace}'", ns);
        }
    }

    private void WriteGlobalEdges(Func<List<GraphEdge>> provider)
    {
        try
        {
            var edges = provider();
            var json = JsonSerializer.Serialize(edges, JsonOptions);
            var path = Path.Combine(_basePath, "_edges.json");
            AtomicWriteAllText(path, json);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to save global edges");
        }
    }

    private void WriteClusters(Func<List<SemanticCluster>> provider)
    {
        try
        {
            var clusters = provider();
            var json = JsonSerializer.Serialize(clusters, JsonOptions);
            var path = Path.Combine(_basePath, "_clusters.json");
            AtomicWriteAllText(path, json);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to save clusters");
        }
    }

    /// <summary>Write to a temp file then rename for crash-safe atomic writes.</summary>
    private static void AtomicWriteAllText(string path, string content)
    {
        var tmpPath = path + ".tmp";
        File.WriteAllText(tmpPath, content);
        File.Move(tmpPath, path, overwrite: true);
    }
}
