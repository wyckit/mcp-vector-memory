using System.Numerics;
using McpVectorMemory.Models;

namespace McpVectorMemory.Services;

/// <summary>
/// Thread-safe namespace-partitioned vector index with lifecycle awareness.
/// Replaces VectorIndex with cognitive memory capabilities.
///
/// Locking strategy:
/// - Read-only methods use EnterUpgradeableReadLock, upgrading to write only if EnsureNamespaceLoaded needs to load.
/// - Mutating methods use EnterWriteLock directly.
/// - Only one upgradeable-read or write lock can be held at a time, but multiple read locks can coexist.
/// </summary>
public sealed class CognitiveIndex : IDisposable
{
    private readonly Dictionary<string, Dictionary<string, (CognitiveEntry Entry, float Norm)>> _namespaces = new();
    private readonly ReaderWriterLockSlim _lock = new();
    private readonly PersistenceManager _persistence;
    private readonly HashSet<string> _loadedNamespaces = new();

    public CognitiveIndex(PersistenceManager persistence)
    {
        _persistence = persistence;
    }

    /// <summary>Total entry count across all loaded namespaces.</summary>
    public int Count
    {
        get
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                LoadAllNamespaces();
                return _namespaces.Values.Sum(ns => ns.Count);
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
            EnsureNamespaceLoaded(ns);
            return _namespaces.TryGetValue(ns, out var entries) ? entries.Count : 0;
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get all known namespace names.</summary>
    public IReadOnlyList<string> GetNamespaces()
    {
        _lock.EnterReadLock();
        try
        {
            var persisted = _persistence.GetPersistedNamespaces();
            var inMemory = _namespaces.Keys;
            return persisted.Union(inMemory).Distinct().ToList();
        }
        finally { _lock.ExitReadLock(); }
    }

    /// <summary>Add or replace a cognitive entry.</summary>
    public void Upsert(CognitiveEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        float norm = Norm(entry.Vector);

        _lock.EnterWriteLock();
        try
        {
            EnsureNamespaceLoadedUnderWrite(entry.Ns);
            if (!_namespaces.TryGetValue(entry.Ns, out var nsEntries))
            {
                nsEntries = new();
                _namespaces[entry.Ns] = nsEntries;
            }
            nsEntries[entry.Id] = (entry, norm);
            ScheduleSave(entry.Ns);
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Get an entry by ID, searching all namespaces.</summary>
    public CognitiveEntry? Get(string id)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            LoadAllNamespaces();
            foreach (var ns in _namespaces.Values)
            {
                if (ns.TryGetValue(id, out var tuple))
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
            EnsureNamespaceLoaded(ns);
            if (_namespaces.TryGetValue(ns, out var entries) && entries.TryGetValue(id, out var tuple))
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
            LoadAllNamespacesUnderWrite();
            foreach (var (ns, entries) in _namespaces)
            {
                if (entries.Remove(id))
                {
                    ScheduleSave(ns);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Namespace-scoped k-nearest-neighbor search with lifecycle filtering.
    /// </summary>
    public IReadOnlyList<CognitiveSearchResult> Search(
        float[] query,
        string ns,
        int k = 5,
        float minScore = 0f,
        string? category = null,
        HashSet<string>? includeStates = null,
        bool summaryFirst = false)
    {
        if (query is null || query.Length == 0)
            throw new ArgumentException("Query vector must not be null or empty.", nameof(query));
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be positive.");

        includeStates ??= new HashSet<string> { "stm", "ltm" };
        float queryNorm = Norm(query);
        if (queryNorm == 0f)
            throw new ArgumentException("Query vector must not be zero-magnitude.", nameof(query));

        List<(CognitiveEntry entry, float score)> scored;
        _lock.EnterUpgradeableReadLock();
        try
        {
            EnsureNamespaceLoaded(ns);
            if (!_namespaces.TryGetValue(ns, out var nsEntries))
                return Array.Empty<CognitiveSearchResult>();

            scored = new(nsEntries.Count);
            foreach (var (entry, entryNorm) in nsEntries.Values)
            {
                if (!includeStates.Contains(entry.LifecycleState))
                    continue;
                if (category is not null && entry.Category != category)
                    continue;
                if (entry.Vector.Length != query.Length)
                    continue;
                if (entryNorm == 0f)
                    continue;

                float dot = Dot(query, entry.Vector);
                float score = dot / (queryNorm * entryNorm);

                if (score >= minScore)
                    scored.Add((entry, score));
            }
        }
        finally { _lock.ExitUpgradeableReadLock(); }

        // Sorting happens outside the lock on a local list
        if (summaryFirst)
        {
            scored.Sort((a, b) =>
            {
                if (a.entry.IsSummaryNode != b.entry.IsSummaryNode)
                    return a.entry.IsSummaryNode ? -1 : 1;
                return b.score.CompareTo(a.score);
            });
        }
        else
        {
            scored.Sort((a, b) => b.score.CompareTo(a.score));
        }

        int take = Math.Min(k, scored.Count);
        var results = new CognitiveSearchResult[take];
        for (int i = 0; i < take; i++)
        {
            var e = scored[i].entry;
            results[i] = new CognitiveSearchResult(
                e.Id, e.Text, scored[i].score, e.LifecycleState,
                e.ActivationEnergy, e.Category, e.Metadata,
                e.IsSummaryNode, e.SourceClusterId, e.AccessCount);
        }
        return results;
    }

    /// <summary>
    /// Search ALL states including archived (for deep_recall).
    /// </summary>
    public IReadOnlyList<CognitiveSearchResult> SearchAllStates(
        float[] query, string ns, int k = 10, float minScore = 0.3f)
    {
        return Search(query, ns, k, minScore,
            includeStates: new HashSet<string> { "stm", "ltm", "archived" });
    }

    /// <summary>Record an access (increments count and updates timestamp).</summary>
    public void RecordAccess(string id)
    {
        _lock.EnterWriteLock();
        try
        {
            LoadAllNamespacesUnderWrite();
            foreach (var (ns, entries) in _namespaces)
            {
                if (entries.TryGetValue(id, out var tuple))
                {
                    tuple.Entry.AccessCount++;
                    tuple.Entry.LastAccessedAt = DateTimeOffset.UtcNow;
                    ScheduleSave(ns);
                    return;
                }
            }
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Update an entry's lifecycle state.</summary>
    public bool SetLifecycleState(string id, string state)
    {
        _lock.EnterWriteLock();
        try
        {
            LoadAllNamespacesUnderWrite();
            foreach (var (ns, entries) in _namespaces)
            {
                if (entries.TryGetValue(id, out var tuple))
                {
                    tuple.Entry.LifecycleState = state;
                    ScheduleSave(ns);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Update activation energy and lifecycle state for an entry atomically.
    /// Used by LifecycleEngine to avoid unsynchronized mutation.
    /// </summary>
    public bool SetActivationEnergyAndState(string id, float activationEnergy, string? newState = null)
    {
        _lock.EnterWriteLock();
        try
        {
            LoadAllNamespacesUnderWrite();
            foreach (var (ns, entries) in _namespaces)
            {
                if (entries.TryGetValue(id, out var tuple))
                {
                    tuple.Entry.ActivationEnergy = activationEnergy;
                    if (newState is not null)
                        tuple.Entry.LifecycleState = newState;
                    ScheduleSave(ns);
                    return true;
                }
            }
            return false;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>Get all entries in a namespace (for lifecycle processing).</summary>
    public IReadOnlyList<CognitiveEntry> GetAllInNamespace(string ns)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            EnsureNamespaceLoaded(ns);
            if (!_namespaces.TryGetValue(ns, out var entries))
                return Array.Empty<CognitiveEntry>();
            return entries.Values.Select(t => t.Entry).ToList();
        }
        finally { _lock.ExitUpgradeableReadLock(); }
    }

    /// <summary>Get all entries across all namespaces.</summary>
    public IReadOnlyList<CognitiveEntry> GetAll()
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            LoadAllNamespaces();
            return _namespaces.Values
                .SelectMany(ns => ns.Values.Select(t => t.Entry))
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
                LoadAllNamespaces();
            else
                EnsureNamespaceLoaded(ns);

            var entries = ns is null || ns == "*"
                ? _namespaces.Values.SelectMany(n => n.Values.Select(t => t.Entry))
                : _namespaces.TryGetValue(ns, out var nsEntries)
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

    public void Dispose()
    {
        _lock.Dispose();
    }

    // Load a namespace from disk if not already loaded.
    // Called under an upgradeable read lock — upgrades to write lock only if loading is needed.
    private void EnsureNamespaceLoaded(string ns)
    {
        if (_loadedNamespaces.Contains(ns))
            return;

        // Upgrade to write lock for the actual mutation
        _lock.EnterWriteLock();
        try
        {
            // Double-check after acquiring write lock
            if (_loadedNamespaces.Contains(ns))
                return;

            var data = _persistence.LoadNamespace(ns);
            if (!_namespaces.ContainsKey(ns))
                _namespaces[ns] = new();

            foreach (var entry in data.Entries)
            {
                float norm = Norm(entry.Vector);
                _namespaces[ns][entry.Id] = (entry, norm);
            }

            _loadedNamespaces.Add(ns);
        }
        finally { _lock.ExitWriteLock(); }
    }

    // Load a namespace from disk if not already loaded.
    // Called when already holding a write lock.
    private void EnsureNamespaceLoadedUnderWrite(string ns)
    {
        if (_loadedNamespaces.Contains(ns))
            return;

        var data = _persistence.LoadNamespace(ns);
        if (!_namespaces.ContainsKey(ns))
            _namespaces[ns] = new();

        foreach (var entry in data.Entries)
        {
            float norm = Norm(entry.Vector);
            _namespaces[ns][entry.Id] = (entry, norm);
        }

        _loadedNamespaces.Add(ns);
    }

    // Load all namespaces. Called under upgradeable read lock.
    private void LoadAllNamespaces()
    {
        foreach (var ns in _persistence.GetPersistedNamespaces())
            EnsureNamespaceLoaded(ns);
    }

    // Load all namespaces. Called when already holding write lock.
    private void LoadAllNamespacesUnderWrite()
    {
        foreach (var ns in _persistence.GetPersistedNamespaces())
            EnsureNamespaceLoadedUnderWrite(ns);
    }

    // Snapshot current namespace data and schedule a debounced write.
    // MUST be called within a write lock (data is captured immediately).
    private void ScheduleSave(string ns)
    {
        // Capture snapshot now while we hold the write lock — no lock re-entry needed
        var data = new NamespaceData();
        if (_namespaces.TryGetValue(ns, out var entries))
            data.Entries = entries.Values.Select(t => t.Entry).ToList();

        _persistence.ScheduleSave(ns, () => data);
    }

    // ── SIMD-accelerated vector math ──

    public static float Dot(float[] a, float[] b)
    {
        float sum = 0f;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            int simdLength = Vector<float>.Count;
            int simdEnd = a.Length - (a.Length % simdLength);
            for (; i < simdEnd; i += simdLength)
                sum += Vector.Dot(new Vector<float>(a, i), new Vector<float>(b, i));
        }

        for (; i < a.Length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    public static float Norm(float[] v)
    {
        float dot = Dot(v, v);
        return dot == 0f ? 0f : MathF.Sqrt(dot);
    }
}
