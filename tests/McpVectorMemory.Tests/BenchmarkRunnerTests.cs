using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using Xunit.Abstractions;

namespace McpVectorMemory.Tests;

public class BenchmarkRunnerTests
{
    private readonly ITestOutputHelper _output;

    public BenchmarkRunnerTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void RecallAtK_AllRelevantRetrieved_Returns1()
    {
        var retrieved = new[] { "a", "b", "c" };
        var relevant = new HashSet<string> { "a", "b", "c" };
        Assert.Equal(1f, BenchmarkRunner.ComputeRecallAtK(retrieved, relevant));
    }

    [Fact]
    public void RecallAtK_NoneRetrieved_Returns0()
    {
        var retrieved = new[] { "x", "y" };
        var relevant = new HashSet<string> { "a", "b" };
        Assert.Equal(0f, BenchmarkRunner.ComputeRecallAtK(retrieved, relevant));
    }

    [Fact]
    public void RecallAtK_PartialRetrieval()
    {
        var retrieved = new[] { "a", "x", "b" };
        var relevant = new HashSet<string> { "a", "b", "c", "d" };
        Assert.Equal(0.5f, BenchmarkRunner.ComputeRecallAtK(retrieved, relevant));
    }

    [Fact]
    public void RecallAtK_EmptyRelevant_Returns1()
    {
        var retrieved = new[] { "a" };
        var relevant = new HashSet<string>();
        Assert.Equal(1f, BenchmarkRunner.ComputeRecallAtK(retrieved, relevant));
    }

    [Fact]
    public void PrecisionAtK_AllRelevant()
    {
        var retrieved = new[] { "a", "b", "c" };
        var relevant = new HashSet<string> { "a", "b", "c" };
        Assert.Equal(1f, BenchmarkRunner.ComputePrecisionAtK(retrieved, relevant, 3));
    }

    [Fact]
    public void PrecisionAtK_HalfRelevant()
    {
        var retrieved = new[] { "a", "x", "b", "y" };
        var relevant = new HashSet<string> { "a", "b" };
        Assert.Equal(0.5f, BenchmarkRunner.ComputePrecisionAtK(retrieved, relevant, 4));
    }

    [Fact]
    public void PrecisionAtK_ZeroK_Returns0()
    {
        var retrieved = new[] { "a" };
        var relevant = new HashSet<string> { "a" };
        Assert.Equal(0f, BenchmarkRunner.ComputePrecisionAtK(retrieved, relevant, 0));
    }

    [Fact]
    public void MRR_FirstResult()
    {
        var retrieved = new[] { "a", "b", "c" };
        var relevant = new HashSet<string> { "a" };
        Assert.Equal(1f, BenchmarkRunner.ComputeMRR(retrieved, relevant));
    }

    [Fact]
    public void MRR_SecondResult()
    {
        var retrieved = new[] { "x", "a", "b" };
        var relevant = new HashSet<string> { "a" };
        Assert.Equal(0.5f, BenchmarkRunner.ComputeMRR(retrieved, relevant));
    }

    [Fact]
    public void MRR_ThirdResult()
    {
        var retrieved = new[] { "x", "y", "a" };
        var relevant = new HashSet<string> { "a" };
        Assert.Equal(1f / 3f, BenchmarkRunner.ComputeMRR(retrieved, relevant), 4);
    }

    [Fact]
    public void MRR_NoRelevant_Returns0()
    {
        var retrieved = new[] { "x", "y" };
        var relevant = new HashSet<string> { "a" };
        Assert.Equal(0f, BenchmarkRunner.ComputeMRR(retrieved, relevant));
    }

    [Fact]
    public void NdcgAtK_PerfectRanking_Returns1()
    {
        var grades = new Dictionary<string, int> { ["a"] = 3, ["b"] = 2, ["c"] = 1 };
        var retrieved = new[] { "a", "b", "c" };
        Assert.Equal(1f, BenchmarkRunner.ComputeNdcgAtK(retrieved, grades, 3), 4);
    }

    [Fact]
    public void NdcgAtK_ReversedRanking_LessThan1()
    {
        var grades = new Dictionary<string, int> { ["a"] = 3, ["b"] = 2, ["c"] = 1 };
        var retrieved = new[] { "c", "b", "a" };
        var ndcg = BenchmarkRunner.ComputeNdcgAtK(retrieved, grades, 3);
        Assert.True(ndcg < 1f);
        Assert.True(ndcg > 0f);
    }

    [Fact]
    public void NdcgAtK_NoRelevantResults_Returns0()
    {
        var grades = new Dictionary<string, int> { ["a"] = 3 };
        var retrieved = new[] { "x", "y", "z" };
        Assert.Equal(0f, BenchmarkRunner.ComputeNdcgAtK(retrieved, grades, 3));
    }

    [Fact]
    public void NdcgAtK_EmptyGrades_Returns0()
    {
        var grades = new Dictionary<string, int>();
        var retrieved = new[] { "a", "b" };
        Assert.Equal(0f, BenchmarkRunner.ComputeNdcgAtK(retrieved, grades, 2));
    }

    [Fact]
    public void DefaultDataset_Has25SeedsAnd20Queries()
    {
        var dataset = BenchmarkRunner.CreateDefaultDataset();
        Assert.Equal(25, dataset.SeedEntries.Count);
        Assert.Equal(20, dataset.Queries.Count);
        Assert.Equal("default-v1", dataset.DatasetId);
    }

    [Fact]
    public void DefaultDataset_AllQueryReferencesExistInSeeds()
    {
        var dataset = BenchmarkRunner.CreateDefaultDataset();
        var seedIds = dataset.SeedEntries.Select(s => s.Id).ToHashSet();
        foreach (var query in dataset.Queries)
        {
            foreach (var gradeId in query.RelevanceGrades.Keys)
                Assert.Contains(gradeId, seedIds);
        }
    }

    [Fact]
    public void DefaultDataset_AllQueriesHaveValidRelevanceGrades()
    {
        var dataset = BenchmarkRunner.CreateDefaultDataset();
        foreach (var query in dataset.Queries)
        {
            Assert.NotEmpty(query.RelevanceGrades);
            Assert.True(query.RelevanceGrades.Values.All(v => v >= 0 && v <= 3));
        }
    }

    [Fact]
    public void DefaultDataset_SeedIdsAreUnique()
    {
        var dataset = BenchmarkRunner.CreateDefaultDataset();
        var ids = dataset.SeedEntries.Select(s => s.Id).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Fact]
    public void DefaultDataset_QueryIdsAreUnique()
    {
        var dataset = BenchmarkRunner.CreateDefaultDataset();
        var ids = dataset.Queries.Select(q => q.QueryId).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Theory]
    [InlineData("paraphrase-v1")]
    [InlineData("multihop-v1")]
    [InlineData("scale-v1")]
    public void Dataset_SeedAndQueryIdsAreUnique(string datasetId)
    {
        var dataset = BenchmarkRunner.CreateDataset(datasetId);
        Assert.NotNull(dataset);
        var seedIds = dataset.SeedEntries.Select(s => s.Id).ToList();
        Assert.Equal(seedIds.Count, seedIds.Distinct().Count());
        var queryIds = dataset.Queries.Select(q => q.QueryId).ToList();
        Assert.Equal(queryIds.Count, queryIds.Distinct().Count());
    }

    [Theory]
    [InlineData("paraphrase-v1")]
    [InlineData("multihop-v1")]
    [InlineData("scale-v1")]
    public void Dataset_AllRelevanceGradesReferenceValidSeeds(string datasetId)
    {
        var dataset = BenchmarkRunner.CreateDataset(datasetId)!;
        var seedIds = dataset.SeedEntries.Select(s => s.Id).ToHashSet();
        foreach (var query in dataset.Queries)
        {
            foreach (var grade in query.RelevanceGrades)
                Assert.True(seedIds.Contains(grade.Key),
                    $"Query '{query.QueryId}' references non-existent seed '{grade.Key}'");
        }
    }

    [Fact]
    public void GetAvailableDatasets_ContainsAllFour()
    {
        var ids = BenchmarkRunner.GetAvailableDatasets();
        Assert.Contains("default-v1", ids);
        Assert.Contains("paraphrase-v1", ids);
        Assert.Contains("multihop-v1", ids);
        Assert.Contains("scale-v1", ids);
    }

    [Fact]
    public void CreateDataset_UnknownId_ReturnsNull()
    {
        Assert.Null(BenchmarkRunner.CreateDataset("nonexistent"));
    }

    [Fact]
    public void ParaphraseDataset_Has25Seeds15Queries()
    {
        var ds = BenchmarkRunner.CreateParaphraseDataset();
        Assert.Equal(25, ds.SeedEntries.Count);
        Assert.Equal(15, ds.Queries.Count);
    }

    [Fact]
    public void MultihopDataset_Has25Seeds15Queries()
    {
        var ds = BenchmarkRunner.CreateMultiHopDataset();
        Assert.Equal(25, ds.SeedEntries.Count);
        Assert.Equal(15, ds.Queries.Count);
    }

    [Fact]
    public void ScaleDataset_Has80Seeds30Queries()
    {
        var ds = BenchmarkRunner.CreateScaleDataset();
        Assert.Equal(80, ds.SeedEntries.Count);
        Assert.Equal(30, ds.Queries.Count);
    }

    [Theory]
    [InlineData("default-v1")]
    [InlineData("paraphrase-v1")]
    [InlineData("multihop-v1")]
    [InlineData("scale-v1")]
    public void RunAllBenchmarks_OutputResults(string datasetId)
    {
        var persistence = new PersistenceManager(Path.Combine(Path.GetTempPath(), $"bench_{Guid.NewGuid():N}"));
        var embedding = new HashEmbeddingService();
        var index = new CognitiveIndex(persistence);
        var runner = new BenchmarkRunner(index, embedding);

        var dataset = BenchmarkRunner.CreateDataset(datasetId)!;
        var result = runner.Run(dataset);

        _output.WriteLine($"=== {datasetId} ===");
        _output.WriteLine($"  Seeds: {result.TotalEntries}, Queries: {result.TotalQueries}");
        _output.WriteLine($"  Recall@K:    {result.MeanRecallAtK:F3}");
        _output.WriteLine($"  Precision@K: {result.MeanPrecisionAtK:F3}");
        _output.WriteLine($"  MRR:         {result.MeanMRR:F3}");
        _output.WriteLine($"  nDCG@K:      {result.MeanNdcgAtK:F3}");
        _output.WriteLine($"  Latency:     {result.MeanLatencyMs:F3}ms (P95: {result.P95LatencyMs:F3}ms)");

        Assert.True(result.MeanRecallAtK >= 0f);
        Assert.True(result.MeanMRR >= 0f);
        Assert.Equal(dataset.SeedEntries.Count, result.TotalEntries);
        Assert.Equal(dataset.Queries.Count, result.TotalQueries);

        index.Dispose();
    }
}
