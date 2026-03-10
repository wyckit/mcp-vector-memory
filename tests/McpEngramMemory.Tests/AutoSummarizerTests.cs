using McpEngramMemory.Core.Models;
using McpEngramMemory.Core.Services;
using McpEngramMemory.Core.Services.Intelligence;

namespace McpEngramMemory.Tests;

public class AutoSummarizerTests
{
    private readonly HashEmbeddingService _embedding = new();

    private CognitiveEntry MakeEntry(string id, string text, string category = "pattern")
    {
        var vector = _embedding.Embed(text);
        return new CognitiveEntry(id, vector, "test", text, category, lifecycleState: "ltm");
    }

    [Fact]
    public void GenerateSummary_EmptyList_ReturnsEmptyCluster()
    {
        var result = AutoSummarizer.GenerateSummary(Array.Empty<CognitiveEntry>());
        Assert.Equal("Empty cluster.", result);
    }

    [Fact]
    public void GenerateSummary_SingleEntry_IncludesText()
    {
        var entries = new[] { MakeEntry("e1", "SIMD-accelerated vector math operations for fast cosine similarity") };
        var result = AutoSummarizer.GenerateSummary(entries);

        Assert.Contains("1 related memories", result);
        Assert.Contains("SIMD", result);
    }

    [Fact]
    public void GenerateSummary_MultipleEntries_ExtractsKeywords()
    {
        var entries = new[]
        {
            MakeEntry("e1", "SIMD acceleration for vector dot product computation"),
            MakeEntry("e2", "Vector quantization using Int8 for memory compression"),
            MakeEntry("e3", "SIMD optimized vector norm calculation with hardware intrinsics"),
        };

        var result = AutoSummarizer.GenerateSummary(entries);

        Assert.Contains("3 related memories", result);
        // "vector" and "simd" should appear as top keywords
        Assert.Contains("vector", result.ToLowerInvariant());
    }

    [Fact]
    public void GenerateSummary_IncludesCategoryBreakdown()
    {
        var entries = new[]
        {
            MakeEntry("e1", "Architecture decision for storage layer", "architecture"),
            MakeEntry("e2", "Architecture decision for search pipeline", "architecture"),
            MakeEntry("e3", "Bug fix for DLL lock issue", "bug-fix"),
        };

        var result = AutoSummarizer.GenerateSummary(entries);

        Assert.Contains("Categories:", result);
        Assert.Contains("architecture", result);
    }

    [Fact]
    public void GenerateSummary_IncludesRepresentativeSnippets()
    {
        var entries = new[]
        {
            MakeEntry("e1", "Short"),
            MakeEntry("e2", "This is a much longer text about the BM25 keyword indexing implementation and its integration with vector search for hybrid retrieval using reciprocal rank fusion"),
            MakeEntry("e3", "Medium length text about search patterns"),
        };

        var result = AutoSummarizer.GenerateSummary(entries);

        Assert.Contains("Key entries:", result);
        // The longest text should appear (possibly truncated)
        Assert.Contains("BM25", result);
    }

    [Fact]
    public void ExtractKeywords_FiltersShortTokens()
    {
        var entries = new[]
        {
            MakeEntry("e1", "I am a small test of keyword extraction"),
            MakeEntry("e2", "I am another small test of keyword extraction"),
        };

        var keywords = AutoSummarizer.ExtractKeywords(entries);

        // Tokens shorter than 3 chars should be filtered
        Assert.DoesNotContain("am", keywords);
        Assert.DoesNotContain("of", keywords);
    }

    [Fact]
    public void ExtractKeywords_RequiresMinFrequency()
    {
        var entries = new[]
        {
            MakeEntry("e1", "unique_word_only_here vector search"),
            MakeEntry("e2", "vector search hybrid retrieval"),
        };

        var keywords = AutoSummarizer.ExtractKeywords(entries);

        // "vector" and "search" appear twice, should be included
        Assert.Contains("vector", keywords);
        Assert.Contains("search", keywords);
    }

    [Fact]
    public void ExtractKeywords_ReturnsMaxKeywords()
    {
        var entries = new[]
        {
            MakeEntry("e1", "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"),
            MakeEntry("e2", "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"),
        };

        var keywords = AutoSummarizer.ExtractKeywords(entries);
        Assert.True(keywords.Count <= 8); // MaxKeywords = 8
    }

    [Fact]
    public void GenerateSummary_TruncatesLongSnippets()
    {
        var longText = new string('x', 200) + " end marker";
        var entries = new[] { MakeEntry("e1", longText) };

        var result = AutoSummarizer.GenerateSummary(entries);

        // Should be truncated with "..."
        Assert.Contains("...", result);
        Assert.DoesNotContain("end marker", result);
    }
}
