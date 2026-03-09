using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using McpVectorMemory.Core.Services.Storage;
using Microsoft.Data.Sqlite;

namespace McpVectorMemory.Tests;

public class SqliteStorageProviderTests : IDisposable
{
    private readonly string _testDbPath;
    private readonly SqliteStorageProvider _provider;

    public SqliteStorageProviderTests()
    {
        _testDbPath = Path.Combine(Path.GetTempPath(), $"sqlite_test_{Guid.NewGuid():N}", "memory.db");
        _provider = new SqliteStorageProvider(_testDbPath, debounceMs: 10);
    }

    public void Dispose()
    {
        _provider.Dispose();
        // Clear SQLite connection pool to release file locks before cleanup
        SqliteConnection.ClearAllPools();
        var dir = Path.GetDirectoryName(_testDbPath);
        if (dir is not null && Directory.Exists(dir))
            Directory.Delete(dir, true);
    }

    [Fact]
    public void LoadNamespace_Empty_ReturnsEmptyData()
    {
        var data = _provider.LoadNamespace("nonexistent");
        Assert.Empty(data.Entries);
    }

    [Fact]
    public void SaveAndLoad_RoundTrips()
    {
        var entry = new CognitiveEntry("test-1", new[] { 1f, 2f, 3f }, "myns", "hello world");
        var data = new NamespaceData { Entries = new List<CognitiveEntry> { entry } };

        _provider.SaveNamespaceSync("myns", data);
        var loaded = _provider.LoadNamespace("myns");

        Assert.Single(loaded.Entries);
        Assert.Equal("test-1", loaded.Entries[0].Id);
        Assert.Equal("hello world", loaded.Entries[0].Text);
        Assert.Equal(new[] { 1f, 2f, 3f }, loaded.Entries[0].Vector);
    }

    [Fact]
    public void SaveNamespaceSync_Overwrites()
    {
        var entry1 = new CognitiveEntry("a", new[] { 1f, 0f }, "ns", "first");
        _provider.SaveNamespaceSync("ns", new NamespaceData { Entries = [entry1] });

        var entry2 = new CognitiveEntry("b", new[] { 0f, 1f }, "ns", "second");
        _provider.SaveNamespaceSync("ns", new NamespaceData { Entries = [entry2] });

        var loaded = _provider.LoadNamespace("ns");
        Assert.Single(loaded.Entries);
        Assert.Equal("b", loaded.Entries[0].Id);
    }

    [Fact]
    public void GetPersistedNamespaces_ListsNamespaces()
    {
        _provider.SaveNamespaceSync("alpha", new NamespaceData { Entries = [new CognitiveEntry("a", new[] { 1f }, "alpha")] });
        _provider.SaveNamespaceSync("beta", new NamespaceData { Entries = [new CognitiveEntry("b", new[] { 1f }, "beta")] });

        var namespaces = _provider.GetPersistedNamespaces();
        Assert.Contains("alpha", namespaces);
        Assert.Contains("beta", namespaces);
    }

    [Fact]
    public void GetPersistedNamespaces_ExcludesUnderscorePrefix()
    {
        _provider.SaveNamespaceSync("_system", new NamespaceData { Entries = [new CognitiveEntry("s", new[] { 1f }, "_system")] });
        _provider.SaveNamespaceSync("normal", new NamespaceData { Entries = [new CognitiveEntry("n", new[] { 1f }, "normal")] });

        var namespaces = _provider.GetPersistedNamespaces();
        Assert.Contains("normal", namespaces);
        Assert.DoesNotContain("_system", namespaces);
    }

    [Fact]
    public void DebouncedSave_FlushesOnDispose()
    {
        var entry = new CognitiveEntry("d1", new[] { 1f, 2f }, "debounce-ns", "debounced");
        var data = new NamespaceData { Entries = [entry] };
        _provider.ScheduleSave("debounce-ns", () => data);

        // Flush forces pending writes
        _provider.Flush();

        var loaded = _provider.LoadNamespace("debounce-ns");
        Assert.Single(loaded.Entries);
        Assert.Equal("d1", loaded.Entries[0].Id);
    }

    [Fact]
    public void GlobalEdges_SaveAndLoad()
    {
        var edges = new List<GraphEdge>
        {
            new("a", "b", "cross_reference"),
            new("b", "c", "depends_on")
        };

        _provider.ScheduleSaveGlobalEdges(() => edges);
        _provider.Flush();

        var loaded = _provider.LoadGlobalEdges();
        Assert.Equal(2, loaded.Count);
        Assert.Equal("a", loaded[0].SourceId);
        Assert.Equal("depends_on", loaded[1].Relation);
    }

    [Fact]
    public void Clusters_SaveAndLoad()
    {
        var clusters = new List<SemanticCluster>
        {
            new("c1", "test", new List<string> { "m1", "m2" }, "test cluster")
        };

        _provider.ScheduleSaveClusters(() => clusters);
        _provider.Flush();

        var loaded = _provider.LoadClusters();
        Assert.Single(loaded);
        Assert.Equal("c1", loaded[0].ClusterId);
    }

    [Fact]
    public void CollapseHistory_SaveAndLoad()
    {
        var records = new List<CollapseRecord>
        {
            new("collapse-1", "c1", "summary-1", "test",
                new List<string> { "orig-1", "orig-2" },
                new Dictionary<string, string> { ["orig-1"] = "ltm", ["orig-2"] = "ltm" },
                DateTimeOffset.UtcNow)
        };

        _provider.ScheduleSaveCollapseHistory(() => records);
        _provider.Flush();

        var loaded = _provider.LoadCollapseHistory();
        Assert.Single(loaded);
        Assert.Equal("c1", loaded[0].ClusterId);
    }

    [Fact]
    public void DecayConfigs_SaveAndLoad()
    {
        var configs = new Dictionary<string, DecayConfig>
        {
            ["test"] = new("test", decayRate: 0.5f)
        };

        _provider.ScheduleSaveDecayConfigs(() => configs);
        _provider.Flush();

        var loaded = _provider.LoadDecayConfigs();
        Assert.Single(loaded);
        Assert.Equal(0.5f, loaded["test"].DecayRate);
    }

    [Fact]
    public void IntegrationWithCognitiveIndex_BasicOperations()
    {
        using var index = new CognitiveIndex(_provider);

        var entry = new CognitiveEntry("idx-1", new[] { 1f, 0f }, "test", "hello");
        index.Upsert(entry);

        var retrieved = index.Get("idx-1", "test");
        Assert.NotNull(retrieved);
        Assert.Equal("hello", retrieved.Text);
    }

    [Fact]
    public void IntegrationWithCognitiveIndex_SearchWorks()
    {
        using var index = new CognitiveIndex(_provider);

        index.Upsert(new CognitiveEntry("s1", new[] { 1f, 0f }, "test", "alpha"));
        index.Upsert(new CognitiveEntry("s2", new[] { 0f, 1f }, "test", "beta"));

        var results = index.Search(new[] { 1f, 0f }, "test", k: 1);
        Assert.Single(results);
        Assert.Equal("s1", results[0].Id);
    }

    [Fact]
    public void IntegrationWithCognitiveIndex_PersistsAcrossInstances()
    {
        // Save with first instance
        using (var index = new CognitiveIndex(_provider))
        {
            index.Upsert(new CognitiveEntry("p1", new[] { 1f, 0f }, "persist", "persisted entry"));
            _provider.Flush();
        }

        // Load with new provider pointing to same DB
        using var provider2 = new SqliteStorageProvider(_testDbPath, debounceMs: 10);
        using var index2 = new CognitiveIndex(provider2);

        var entry = index2.Get("p1", "persist");
        Assert.NotNull(entry);
        Assert.Equal("persisted entry", entry.Text);
    }

    [Fact]
    public void MultipleEntries_PreservesAll()
    {
        var entries = Enumerable.Range(1, 20).Select(i =>
            new CognitiveEntry($"multi-{i}", new[] { (float)i, 0f }, "multi", $"entry {i}")).ToList();

        _provider.SaveNamespaceSync("multi", new NamespaceData { Entries = entries });
        var loaded = _provider.LoadNamespace("multi");

        Assert.Equal(20, loaded.Entries.Count);
    }

    [Fact]
    public void StorageVersion_IsSet()
    {
        var entry = new CognitiveEntry("v1", new[] { 1f }, "ver", "versioned");
        _provider.SaveNamespaceSync("ver", new NamespaceData { Entries = [entry] });

        var loaded = _provider.LoadNamespace("ver");
        Assert.Equal(1, loaded.StorageVersion);
    }
}
