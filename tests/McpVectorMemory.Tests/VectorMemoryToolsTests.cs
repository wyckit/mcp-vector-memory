using McpVectorMemory;

namespace McpVectorMemory.Tests;

public class VectorMemoryToolsTests
{
    private readonly VectorIndex _index = new();
    private readonly VectorMemoryTools _tools;

    public VectorMemoryToolsTests()
    {
        _tools = new VectorMemoryTools(_index);
    }

    // ── StoreMemory ──────────────────────────────────────────────────────────

    [Fact]
    public void StoreMemory_ValidInput_StoresAndReturnsMessage()
    {
        string result = _tools.StoreMemory("test1", new float[] { 1f, 0f }, "hello");
        Assert.Contains("test1", result);
        Assert.Contains("2-dim", result);
        Assert.Equal(1, _index.Count);
    }

    [Fact]
    public void StoreMemory_WithMetadata_StoresSuccessfully()
    {
        var metadata = new Dictionary<string, string> { ["source"] = "test" };
        string result = _tools.StoreMemory("m1", new float[] { 1f, 2f }, "text", metadata);
        Assert.Contains("m1", result);
        Assert.Equal(1, _index.Count);
    }

    [Fact]
    public void StoreMemory_EmptyId_ReturnsError()
    {
        string result = _tools.StoreMemory("", new float[] { 1f, 0f });
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void StoreMemory_EmptyVector_ReturnsError()
    {
        string result = _tools.StoreMemory("test1", Array.Empty<float>());
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void StoreMemory_SameId_Replaces()
    {
        _tools.StoreMemory("a", new float[] { 1f, 0f }, "first");
        _tools.StoreMemory("a", new float[] { 0f, 1f }, "second");
        Assert.Equal(1, _index.Count);
    }

    [Fact]
    public void StoreMemory_ZeroMagnitudeVector_ReturnsError()
    {
        string result = _tools.StoreMemory("a", new float[] { 0f, 0f, 0f });
        Assert.StartsWith("Error:", result);
        Assert.Equal(0, _index.Count);
    }

    // ── SearchMemory ─────────────────────────────────────────────────────────

    [Fact]
    public void SearchMemory_ReturnsResults()
    {
        _tools.StoreMemory("a", new float[] { 1f, 0f }, "first");
        _tools.StoreMemory("b", new float[] { 0f, 1f }, "second");

        var result = _tools.SearchMemory(new float[] { 1f, 0f }, k: 1);
        var results = Assert.IsType<SearchResult[]>(result);
        Assert.Single(results);
        Assert.Equal("a", results[0].Entry.Id);
    }

    [Fact]
    public void SearchMemory_EmptyIndex_ReturnsEmpty()
    {
        var result = _tools.SearchMemory(new float[] { 1f, 0f });
        var results = Assert.IsType<SearchResult[]>(result);
        Assert.Empty(results);
    }

    [Fact]
    public void SearchMemory_ZeroMagnitudeVector_ReturnsError()
    {
        _tools.StoreMemory("a", new float[] { 1f, 0f });
        var result = _tools.SearchMemory(new float[] { 0f, 0f });
        Assert.IsType<string>(result);
        Assert.StartsWith("Error:", (string)result);
    }

    [Fact]
    public void SearchMemory_ZeroK_ReturnsError()
    {
        var result = _tools.SearchMemory(new float[] { 1f, 0f }, k: 0);
        Assert.IsType<string>(result);
        Assert.StartsWith("Error:", (string)result);
    }

    [Fact]
    public void SearchMemory_MinScoreOutOfRange_ReturnsError()
    {
        var result = _tools.SearchMemory(new float[] { 1f, 0f }, minScore: 1.5f);
        Assert.IsType<string>(result);
        Assert.StartsWith("Error:", (string)result);
    }

    // ── DeleteMemory ─────────────────────────────────────────────────────────

    [Fact]
    public void DeleteMemory_Existing_ReturnsDeleted()
    {
        _tools.StoreMemory("a", new float[] { 1f, 0f });
        string result = _tools.DeleteMemory("a");
        Assert.Contains("Deleted", result);
        Assert.Equal(0, _index.Count);
    }

    [Fact]
    public void DeleteMemory_NonExistent_ReturnsNotFound()
    {
        string result = _tools.DeleteMemory("missing");
        Assert.Contains("not found", result);
    }

    // ── StoreMemories (bulk) ────────────────────────────────────────────────

    [Fact]
    public void StoreMemories_MultipleEntries_Stores()
    {
        var entries = new[]
        {
            new MemoryInput { Id = "a", Vector = new float[] { 1f, 0f }, Text = "alpha" },
            new MemoryInput { Id = "b", Vector = new float[] { 0f, 1f }, Text = "beta" },
        };
        string result = _tools.StoreMemories(entries);
        Assert.Contains("2", result);
        Assert.Equal(2, _index.Count);
    }

    [Fact]
    public void StoreMemories_EmptyArray_ReturnsError()
    {
        string result = _tools.StoreMemories(Array.Empty<MemoryInput>());
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void StoreMemories_NullArray_ReturnsError()
    {
        string result = _tools.StoreMemories(null!);
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void StoreMemories_InvalidEntry_ReturnsError()
    {
        var entries = new[]
        {
            new MemoryInput { Id = "", Vector = new float[] { 1f } }, // invalid: empty id
        };
        string result = _tools.StoreMemories(entries);
        Assert.StartsWith("Error:", result);
    }

    // ── DeleteMemories (bulk) ────────────────────────────────────────────────

    [Fact]
    public void DeleteMemories_RemovesEntries()
    {
        _tools.StoreMemory("a", new float[] { 1f, 0f });
        _tools.StoreMemory("b", new float[] { 0f, 1f });
        _tools.StoreMemory("c", new float[] { 0.7071f, 0.7071f });

        string result = _tools.DeleteMemories(new[] { "a", "c" });
        Assert.Contains("2 of 3", result);
        Assert.Equal(1, _index.Count);
    }

    [Fact]
    public void DeleteMemories_EmptyArray_ReturnsError()
    {
        string result = _tools.DeleteMemories(Array.Empty<string>());
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void DeleteMemories_NullArray_ReturnsError()
    {
        string result = _tools.DeleteMemories(null!);
        Assert.StartsWith("Error:", result);
    }

    // ── SearchMemory with offset ─────────────────────────────────────────────

    [Fact]
    public void SearchMemory_WithOffset_SkipsResults()
    {
        _tools.StoreMemory("close", new float[] { 1f, 0.1f });
        _tools.StoreMemory("far",   new float[] { 0f, 1f });

        var result = _tools.SearchMemory(new float[] { 1f, 0f }, k: 1, offset: 1);
        var results = Assert.IsType<SearchResult[]>(result);
        Assert.Single(results);
        Assert.Equal("far", results[0].Entry.Id);
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_NullIndex_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new VectorMemoryTools(null!));
    }
}
