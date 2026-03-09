using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services.Intelligence;
using McpVectorMemory.Core.Services.Retrieval;
using McpVectorMemory.Core.Services.Storage;

namespace McpVectorMemory.Core.Services;

/// <summary>
/// Thread-safe namespace-partitioned vector index with lifecycle awareness.
/// Thin facade that delegates search to VectorSearchEngine/HybridSearchEngine,
/// duplicate detection to DuplicateDetector, and manages CRUD + locking.
/// </summary>
public sealed class CognitiveIndex : IDisposable
{
    private readonly NamespaceStore _store;
    private readonly ReaderWriterLockSlim _lock = new();
    private readonly BM25Index _bm25 = new();
    private readonly TokenReranker _reranker = new();
    private readonly VectorSearchEngine _vectorSearch = new();
    private readonly HybridSearchEngine _hybridSearch = new();
    private readonly DuplicateDetector _duplicateDetector = new();
    private readonly MemoryLimitsConfig _limits;

    public CognitiveIndex(IStorageProvider persistence, MemoryLimitsConfig? limits = null)
    {
        _store = new NamespaceStore(persistence, _bm25);
        _limits = limits ?? new MemoryLimitsConfig();
    }

    // ── Counts + Metadata ──

    /// <summary>Total entry count across all loaded namespaces.</summary>
    public int Count
    {
        get
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                _store.LoadAll();
                return _store.TotalCount;
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }
    }

    /// <summary>Count entries in a specific namespace.</summary>
    public int CountInNamespace(string ns)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            return _store.GetNamespace(ns)?.Count ?? 0;
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get all known namespace names.</summary>
    public IReadOnlyList<string> GetNamespaces()
    {
        _lock.EnterReadLock();
        try { return _store.GetNamespaceNames(); }
        finally { _lock.ExitReadLock(); }
    }

    // ── CRUD ──

    /// <summary>Add or replace a cognitive entry. LTM/archived entries are auto-quantized for fast search.</summary>
    public void Upsert(CognitiveEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        float norm = VectorMath.Norm(entry.Vector);
        var quantized = entry.LifecycleState is "ltm" or "archived"
            ? VectorQuantizer.Quantize(entry.Vector)
            : null;

        _lock.EnterWriteLock();
        try
        {
            _store.EnsureLoaded(entry.Ns);
            var nsEntries = _store.GetOrCreateNamespace(entry.Ns);

            // Enforce memory limits (skip for updates to existing entries)
            if (!nsEntries.ContainsKey(entry.Id))
            {
                if (nsEntries.Count >= _limits.MaxNamespaceSize)
                    throw new InvalidOperationException(
                        $"Namespace '{entry.Ns}' has reached the maximum size of {_limits.MaxNamespaceSize} entries.");
                if (_store.TotalCount >= _limits.MaxTotalCount)
                    throw new InvalidOperationException(
                        $"Total memory count has reached the maximum of {_limits.MaxTotalCount} entries.");
            }

            nsEntries[entry.Id] = (entry, norm, quantized);
            _store.IndexBM25(entry);
            _store.ScheduleSave(entry.Ns);
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Get an entry by ID, searching all namespaces.</summary>
    public CognitiveEntry? Get(string id)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.LoadAll();
            foreach (var kv in _store.AllNamespaces)
            {
                if (kv.Value.TryGetValue(id, out var tuple))
                    return tuple.Entry;
            }
            return null;
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get an entry by ID within a specific namespace.</summary>
    public CognitiveEntry? Get(string id, string ns)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is not null && nsEntries.TryGetValue(id, out var tuple))
                return tuple.Entry;
            return null;
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Delete an entry by ID, searching all namespaces.</summary>
    public bool Delete(string id)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.LoadAll();
            foreach (var kv in _store.AllNamespaces)
            {
                if (kv.Value.Remove(id))
                {
                    _store.RemoveBM25(id, kv.Key);
                    _store.ScheduleSave(kv.Key);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    // ── Search (delegated to VectorSearchEngine) ──

    /// <summary>Namespace-scoped k-nearest-neighbor search with two-stage Int8 screening pipeline.</summary>
    public IReadOnlyList<CognitiveSearchResult> Search(
        float[] query, string ns, int k = 5, float minScore = 0f,
        string? category = null, HashSet<string>? includeStates = null, bool summaryFirst = false)
    {
        // Validate arguments before any locking or short-circuit returns
        if (query is null || query.Length == 0)
            throw new ArgumentException("Query vector must not be null or empty.", nameof(query));
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be positive.");

        IReadOnlyCollection<(CognitiveEntry Entry, float Norm, QuantizedVector? Quantized)> snapshot;
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is null || nsEntries.Count == 0)
                return Array.Empty<CognitiveSearchResult>();
            snapshot = nsEntries.Values.ToList();
        }
        finally { _lock.ExitUpgradeableReadLock(); }

        return _vectorSearch.Search(query, snapshot, k, minScore, category, includeStates, summaryFirst);
    }

    /// <summary>Hybrid search combining vector + BM25 via Reciprocal Rank Fusion.</summary>
    public IReadOnlyList<CognitiveSearchResult> HybridSearch(
        float[] query, string queryText, string ns, int k = 5, float minScore = 0f,
        string? category = null, HashSet<string>? includeStates = null,
        bool rerank = false, int rrfK = 60)
    {
        int candidateK = Math.Max(k * 4, 20);
        var vectorResults = Search(query, ns, candidateK, minScore, category, includeStates);

        return _hybridSearch.HybridSearch(
            vectorResults, queryText, ns, k, includeStates, category,
            rerank, rrfK, _bm25, _reranker, Get);
    }

    /// <summary>Apply token-level reranking to existing search results.</summary>
    public IReadOnlyList<CognitiveSearchResult> Rerank(
        string queryText, IReadOnlyList<CognitiveSearchResult> results)
        => _reranker.Rerank(queryText, results);

    /// <summary>Search ALL states including archived (for deep_recall).</summary>
    public IReadOnlyList<CognitiveSearchResult> SearchAllStates(
        float[] query, string ns, int k = 10, float minScore = 0.3f)
        => Search(query, ns, k, minScore,
            includeStates: new HashSet<string> { "stm", "ltm", "archived" });

    // ── Duplicate Detection (delegated to DuplicateDetector) ──

    /// <summary>Find near-duplicates for a single entry within its namespace (O(N) scan).</summary>
    public IReadOnlyList<(string IdA, string IdB, float Similarity)> FindDuplicatesForEntry(
        string ns, string entryId, float threshold = 0.95f)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is null)
                return Array.Empty<(string, string, float)>();

            nsEntries.TryGetValue(entryId, out var target);
            return _duplicateDetector.FindDuplicatesForEntry(entryId, target, nsEntries, threshold);
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Find near-duplicate entries within a namespace by pairwise cosine similarity.</summary>
    public IReadOnlyList<(string IdA, string IdB, float Similarity)> FindDuplicates(
        string ns, float threshold = 0.95f, string? category = null,
        HashSet<string>? includeStates = null, int maxResults = 100)
    {
        if (threshold < 0f || threshold > 1f)
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 1.");

        includeStates ??= new HashSet<string> { "stm", "ltm" };

        List<(CognitiveEntry Entry, float Norm, QuantizedVector? Quantized)> candidates;
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is null)
                return Array.Empty<(string, string, float)>();

            candidates = nsEntries.Values
                .Where(t => includeStates.Contains(t.Entry.LifecycleState)
                    && (category is null || t.Entry.Category == category)
                    && t.Norm > 0f)
                .ToList();
        }
        finally { _lock.ExitUpgradeableReadLock(); }

        // Sort by norm ascending for early-exit optimization
        candidates.Sort((a, b) => a.Norm.CompareTo(b.Norm));
        return _duplicateDetector.FindDuplicates(candidates, threshold, maxResults);
    }

    // ── Access Tracking ──

    /// <summary>Record an access (increments count and updates timestamp).</summary>
    public void RecordAccess(string id)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.LoadAll();
            foreach (var kv in _store.AllNamespaces)
            {
                if (kv.Value.TryGetValue(id, out var tuple))
                {
                    tuple.Entry.AccessCount++;
                    tuple.Entry.LastAccessedAt = DateTimeOffset.UtcNow;
                    _store.ScheduleSave(kv.Key);
                    return;
                }
            }
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Record an access hit within a known namespace.</summary>
    public void RecordAccess(string id, string ns)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is not null && nsEntries.TryGetValue(id, out var tuple))
            {
                tuple.Entry.AccessCount++;
                tuple.Entry.LastAccessedAt = DateTimeOffset.UtcNow;
                _store.ScheduleSave(ns);
            }
        }
        finally { _lock.ExitWriteLock(); }
    }

    // ── Lifecycle State Management ──

    /// <summary>Update an entry's lifecycle state. Quantizes on STM→LTM, dequantizes on →STM.</summary>
    public bool SetLifecycleState(string id, string state)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.LoadAll();
            foreach (var kv in _store.AllNamespaces)
            {
                if (kv.Value.TryGetValue(id, out var tuple))
                {
                    var previousState = tuple.Entry.LifecycleState;
                    tuple.Entry.LifecycleState = state;
                    UpdateQuantization(kv.Value, id, tuple, previousState, state);
                    _store.ScheduleSave(kv.Key);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Update lifecycle state for multiple entries in a single write lock acquisition.</summary>
    public int SetLifecycleStateBatch(IEnumerable<string> ids, string state)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.LoadAll();
            int updated = 0;
            var dirtied = new HashSet<string>();
            foreach (var id in ids)
            {
                foreach (var kv in _store.AllNamespaces)
                {
                    if (kv.Value.TryGetValue(id, out var tuple))
                    {
                        var previousState = tuple.Entry.LifecycleState;
                        tuple.Entry.LifecycleState = state;
                        UpdateQuantization(kv.Value, id, tuple, previousState, state);
                        dirtied.Add(kv.Key);
                        updated++;
                        break;
                    }
                }
            }
            foreach (var ns in dirtied)
                _store.ScheduleSave(ns);
            return updated;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Update activation energy and lifecycle state atomically.</summary>
    public bool SetActivationEnergyAndState(string id, float activationEnergy, string? newState = null)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.LoadAll();
            foreach (var kv in _store.AllNamespaces)
            {
                if (kv.Value.TryGetValue(id, out var tuple))
                {
                    var previousState = tuple.Entry.LifecycleState;
                    tuple.Entry.ActivationEnergy = activationEnergy;
                    if (newState is not null)
                    {
                        tuple.Entry.LifecycleState = newState;
                        UpdateQuantization(kv.Value, id, tuple, previousState, newState);
                    }
                    _store.ScheduleSave(kv.Key);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    // ── Bulk Reads ──

    /// <summary>Get all entries in a namespace.</summary>
    public IReadOnlyList<CognitiveEntry> GetAllInNamespace(string ns)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is null)
                return Array.Empty<CognitiveEntry>();
            return nsEntries.Values.Select(t => t.Entry).ToList();
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get all entries across all namespaces.</summary>
    public IReadOnlyList<CognitiveEntry> GetAll()
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            _store.LoadAll();
            return _store.AllNamespaces
                .SelectMany(kv => kv.Value.Values.Select(t => t.Entry))
                .ToList();
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get count of entries by lifecycle state.</summary>
    public (int stm, int ltm, int archived) GetStateCounts(string? ns = null)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            if (ns is null || ns == "*")
                _store.LoadAll();
            else
                _store.EnsureLoaded(ns);

            var entries = ns is null || ns == "*"
                ? _store.AllNamespaces.SelectMany(kv => kv.Value.Values.Select(t => t.Entry))
                : _store.GetNamespace(ns) is { } nsEntries
                    ? nsEntries.Values.Select(t => t.Entry)
                    : Enumerable.Empty<CognitiveEntry>();

            int stm = 0, ltm = 0, archived = 0;
            foreach (var e in entries)
            {
                switch (e.LifecycleState)
                {
                    case "stm": stm++; break;
                    case "ltm": ltm++; break;
                    case "archived": archived++; break;
                }
            }
            return (stm, ltm, archived);
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Re-embed all entries in a namespace.</summary>
    public (int Updated, int Skipped) RebuildEmbeddings(string ns, IEmbeddingService embedding)
    {
        _lock.EnterWriteLock();
        try
        {
            _store.EnsureLoaded(ns);
            var nsEntries = _store.GetNamespace(ns);
            if (nsEntries is null)
                return (0, 0);

            int updated = 0, skipped = 0;
            var ids = nsEntries.Keys.ToList();

            foreach (var id in ids)
            {
                var (oldEntry, _, _) = nsEntries[id];
                if (string.IsNullOrWhiteSpace(oldEntry.Text))
                {
                    skipped++;
                    continue;
                }

                float[] newVector = embedding.Embed(oldEntry.Text);
                var newEntry = new CognitiveEntry(
                    oldEntry.Id, newVector, oldEntry.Ns, oldEntry.Text,
                    oldEntry.Category, oldEntry.Metadata, oldEntry.LifecycleState,
                    oldEntry.CreatedAt, oldEntry.LastAccessedAt, oldEntry.AccessCount,
                    oldEntry.ActivationEnergy, oldEntry.IsSummaryNode, oldEntry.SourceClusterId);

                var quantized = newEntry.LifecycleState is "ltm" or "archived"
                    ? VectorQuantizer.Quantize(newVector)
                    : null;
                nsEntries[id] = (newEntry, VectorMath.Norm(newVector), quantized);
                _store.IndexBM25(newEntry);
                updated++;
            }

            if (updated > 0)
                _store.ScheduleSave(ns);

            return (updated, skipped);
        }
        finally { _lock.ExitWriteLock(); }
    }

    public void Dispose() => _lock.Dispose();

    // ── Internal Helpers ──

    private static void UpdateQuantization(
        Dictionary<string, (CognitiveEntry Entry, float Norm, QuantizedVector? Quantized)> entries,
        string id,
        (CognitiveEntry Entry, float Norm, QuantizedVector? Quantized) tuple,
        string previousState, string newState)
    {
        bool wasQuantizable = previousState is "ltm" or "archived";
        bool isQuantizable = newState is "ltm" or "archived";

        if (!wasQuantizable && isQuantizable && tuple.Quantized is null)
            entries[id] = (tuple.Entry, tuple.Norm, VectorQuantizer.Quantize(tuple.Entry.Vector));
        else if (wasQuantizable && !isQuantizable && tuple.Quantized is not null)
            entries[id] = (tuple.Entry, tuple.Norm, null);
    }

}
