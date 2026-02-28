namespace McpVectorMemory;

/// <summary>
/// Thread-safe vector index supporting upsert, delete, and k-nearest-neighbor
/// search via cosine similarity. Uses HNSW for sub-linear search and optionally
/// persists entries to disk as JSON.
/// </summary>
public sealed class VectorIndex : IDisposable
{
    // Per-dimension HNSW graphs (vectors of different dimensions live in separate graphs)
    private readonly Dictionary<int, HnswGraph> _graphs = new();

    // String ID → internal integer ID
    private readonly Dictionary<string, int> _idMap = new();

    // Internal ID → (entry, dimension)
    private readonly Dictionary<int, (VectorEntry Entry, int Dim)> _entries = new();

    private readonly ReaderWriterLockSlim _lock = new();
    private readonly string? _dataPath;
    private readonly int _hnswM;
    private readonly int _hnswEfConstruction;
    private readonly int _hnswEfSearch;
    private readonly TimeSpan? _defaultTtl;
    private int _nextId;
    private int _count;
    private int _deletedNodeCount; // tracks soft-deleted HNSW nodes for compaction

    /// <summary>Number of vectors currently stored in the index.</summary>
    public int Count => Volatile.Read(ref _count);

    /// <summary>
    /// Creates a new vector index.
    /// </summary>
    /// <param name="dataPath">
    /// File path for JSON persistence. Pass <c>null</c> for ephemeral in-memory only.
    /// </param>
    /// <param name="hnswM">HNSW M parameter — max connections per node per layer (default 16).</param>
    /// <param name="hnswEfConstruction">HNSW construction search effort (default 200).</param>
    /// <param name="hnswEfSearch">HNSW search effort (default 50).</param>
    /// <param name="defaultTtl">
    /// Optional time-to-live for entries. Expired entries are excluded from search
    /// results and purged on the next mutation. Pass <c>null</c> to disable expiration.
    /// </param>
    public VectorIndex(
        string? dataPath = null,
        int hnswM = 16,
        int hnswEfConstruction = 200,
        int hnswEfSearch = 50,
        TimeSpan? defaultTtl = null)
    {
        if (defaultTtl.HasValue && defaultTtl.Value <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(defaultTtl), "TTL must be positive.");

        _dataPath = dataPath;
        _hnswM = hnswM;
        _hnswEfConstruction = hnswEfConstruction;
        _hnswEfSearch = hnswEfSearch;
        _defaultTtl = defaultTtl;

        if (_dataPath is not null)
            LoadFromDisk();
    }

    /// <summary>
    /// Adds or replaces a vector entry.
    /// </summary>
    public void Upsert(VectorEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);

        _lock.EnterWriteLock();
        try
        {
            UpsertUnsafe(entry);
            EvictExpiredUnsafe();
            RebuildIfNeeded();
            PersistUnsafe();
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Removes a vector entry by id. Returns <c>true</c> if it existed.
    /// </summary>
    public bool Delete(string id)
    {
        _lock.EnterWriteLock();
        try
        {
            if (!DeleteUnsafe(id))
                return false;

            RebuildIfNeeded();
            PersistUnsafe();
            return true;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Adds or replaces multiple vector entries in a single lock acquisition.
    /// More efficient than calling <see cref="Upsert"/> in a loop because the
    /// write lock is held once and persistence is written once at the end.
    /// </summary>
    /// <returns>The number of entries that were inserted or replaced.</returns>
    public int BulkUpsert(IEnumerable<VectorEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        var entryList = entries.ToList();
        if (entryList.Count == 0)
            return 0;

        _lock.EnterWriteLock();
        try
        {
            foreach (var entry in entryList)
            {
                ArgumentNullException.ThrowIfNull(entry);
                UpsertUnsafe(entry);
            }

            EvictExpiredUnsafe();
            RebuildIfNeeded();
            PersistUnsafe();
            return entryList.Count;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Deletes multiple entries by id in a single lock acquisition.
    /// </summary>
    /// <returns>The number of entries that were actually deleted.</returns>
    public int BulkDelete(IEnumerable<string> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var idList = ids.ToList();
        if (idList.Count == 0)
            return 0;

        _lock.EnterWriteLock();
        try
        {
            int deleted = 0;
            foreach (var id in idList)
            {
                if (DeleteUnsafe(id))
                    deleted++;
            }

            if (deleted > 0)
            {
                RebuildIfNeeded();
                PersistUnsafe();
            }
            return deleted;
        }
        finally { _lock.ExitWriteLock(); }
    }

    /// <summary>
    /// Searches for the <paramref name="k"/> nearest neighbors of
    /// <paramref name="query"/> using cosine similarity (higher = more similar).
    /// </summary>
    /// <param name="query">Query vector (must have the same dimension as stored vectors).</param>
    /// <param name="k">Maximum number of results to return.</param>
    /// <param name="minScore">Minimum cosine-similarity threshold (-1 to 1).</param>
    /// <param name="offset">Number of top results to skip (for pagination). Defaults to 0.</param>
    public IReadOnlyList<SearchResult> Search(float[] query, int k = 5, float minScore = 0f, int offset = 0)
    {
        if (query is null || query.Length == 0)
            throw new ArgumentException("Query vector must not be null or empty.", nameof(query));
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be positive.");
        if (float.IsNaN(minScore) || minScore < -1f || minScore > 1f)
            throw new ArgumentOutOfRangeException(nameof(minScore), "minScore must be between -1 and 1.");
        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset), "offset must not be negative.");

        float queryNorm = VectorMath.Norm(query);
        if (queryNorm == 0f)
            throw new ArgumentException("Query vector must not be zero-magnitude.", nameof(query));

        _lock.EnterReadLock();
        try
        {
            int dim = query.Length;
            if (!_graphs.TryGetValue(dim, out var graph))
                return Array.Empty<SearchResult>();

            // Fetch enough candidates to satisfy offset + k
            int needed = offset + k;
            int ef = Math.Max(needed * 2, _hnswEfSearch);
            float maxDistance = 1f - minScore; // cosine distance threshold
            var hnswResults = graph.Search(query, k: ef, ef: ef);

            DateTime? cutoff = _defaultTtl.HasValue ? DateTime.UtcNow - _defaultTtl.Value : null;

            var results = new List<SearchResult>(Math.Min(k, hnswResults.Count));
            int skipped = 0;
            foreach (var (internalId, distance) in hnswResults)
            {
                // HNSW results are sorted by distance ascending — once past the
                // threshold, all remaining results are also too far.
                if (distance > maxDistance)
                    break;

                if (!_entries.TryGetValue(internalId, out var e))
                    continue;

                // Skip expired entries
                if (cutoff.HasValue && e.Entry.CreatedAtUtc < cutoff.Value)
                    continue;

                // Skip entries for pagination offset
                if (skipped < offset)
                {
                    skipped++;
                    continue;
                }

                float score = 1f - distance;
                results.Add(new SearchResult(e.Entry, score));
                if (results.Count >= k)
                    break;
            }

            return results.ToArray();
        }
        finally { _lock.ExitReadLock(); }
    }

    /// <summary>
    /// Returns a snapshot of index statistics for diagnostics and monitoring.
    /// </summary>
    public IndexStatistics GetStatistics()
    {
        _lock.EnterReadLock();
        try
        {
            var dimensions = _graphs.Keys.OrderBy(d => d).ToArray();
            var entriesPerDimension = new Dictionary<int, int>();
            foreach (var (internalId, (_, dim)) in _entries)
            {
                entriesPerDimension.TryGetValue(dim, out int count);
                entriesPerDimension[dim] = count + 1;
            }

            return new IndexStatistics(
                EntryCount: _count,
                PendingDeletions: _deletedNodeCount,
                Dimensions: dimensions,
                EntriesPerDimension: entriesPerDimension,
                IsPersistent: _dataPath is not null);
        }
        finally { _lock.ExitReadLock(); }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _lock.Dispose();
    }

    // ── unsafe helpers (must be called under write lock) ───────────────────

    /// <summary>Upserts a single entry. Must be called under write lock.</summary>
    private void UpsertUnsafe(VectorEntry entry)
    {
        int dim = entry.Vector.Length;

        // Handle replacement: remove old internal mapping
        if (_idMap.TryGetValue(entry.Id, out int oldInternalId))
        {
            var (_, oldDim) = _entries[oldInternalId];
            GetOrCreateGraph(oldDim).MarkDeleted(oldInternalId);
            _entries.Remove(oldInternalId);
            _deletedNodeCount++;
        }
        else
        {
            _count++;
        }

        int internalId = _nextId++;
        _idMap[entry.Id] = internalId;
        _entries[internalId] = (entry, dim);

        var graph = GetOrCreateGraph(dim);
        graph.Add(internalId, entry.Vector);
    }

    /// <summary>Deletes a single entry by id. Must be called under write lock.</summary>
    private bool DeleteUnsafe(string id)
    {
        if (!_idMap.TryGetValue(id, out int internalId))
            return false;

        var (_, dim) = _entries[internalId];
        GetOrCreateGraph(dim).MarkDeleted(internalId);
        _entries.Remove(internalId);
        _idMap.Remove(id);
        _count--;
        _deletedNodeCount++;
        return true;
    }

    /// <summary>
    /// Removes expired entries when TTL is configured. Must be called under write lock.
    /// </summary>
    private void EvictExpiredUnsafe()
    {
        if (!_defaultTtl.HasValue)
            return;

        var cutoff = DateTime.UtcNow - _defaultTtl.Value;
        var expiredIds = new List<string>();
        foreach (var (_, (entry, _)) in _entries)
        {
            if (entry.CreatedAtUtc < cutoff)
                expiredIds.Add(entry.Id);
        }

        foreach (var id in expiredIds)
            DeleteUnsafe(id);
    }

    private HnswGraph GetOrCreateGraph(int dimension)
    {
        if (!_graphs.TryGetValue(dimension, out var graph))
        {
            graph = new HnswGraph(_hnswM, _hnswEfConstruction);
            _graphs[dimension] = graph;
        }
        return graph;
    }

    /// <summary>
    /// Loads entries from disk and rebuilds the HNSW graphs.
    /// Must be called before any concurrent access (i.e. from the constructor).
    /// </summary>
    private void LoadFromDisk()
    {
        var entries = IndexPersistence.Load(_dataPath!);
        foreach (var entry in entries)
        {
            int dim = entry.Vector.Length;
            int internalId = _nextId++;
            _idMap[entry.Id] = internalId;
            _entries[internalId] = (entry, dim);

            var graph = GetOrCreateGraph(dim);
            graph.Add(internalId, entry.Vector);
            _count++;
        }
    }

    /// <summary>
    /// Rebuilds all HNSW graphs from scratch when the number of soft-deleted
    /// nodes exceeds the live count. Compact alone can leave the graph
    /// fragmented (orphaned nodes with no connections at higher layers),
    /// so a full rebuild guarantees optimal graph quality.
    /// Must be called under write lock.
    /// </summary>
    private void RebuildIfNeeded()
    {
        int threshold = Math.Max(_count, 100);
        if (_deletedNodeCount < threshold)
            return;

        _graphs.Clear();
        foreach (var (internalId, (entry, dim)) in _entries)
        {
            var graph = GetOrCreateGraph(dim);
            graph.Add(internalId, entry.Vector);
        }
        _deletedNodeCount = 0;
    }

    /// <summary>
    /// Persists current entries to disk. Must be called under write lock.
    /// </summary>
    private void PersistUnsafe()
    {
        if (_dataPath is null) return;
        IndexPersistence.Save(_dataPath, _entries.Values.Select(e => e.Entry));
    }
}
