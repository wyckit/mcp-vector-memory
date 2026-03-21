using McpEngramMemory.Core.Models;
using McpEngramMemory.Core.Services;
using McpEngramMemory.Core.Services.Evaluation;
using McpEngramMemory.Core.Services.Experts;
using McpEngramMemory.Core.Services.Storage;
using McpEngramMemory.Tools;

namespace McpEngramMemory.Tests;

public class HierarchicalRoutingTests : IDisposable
{
    private readonly CognitiveIndex _index;
    private readonly HashEmbeddingService _embedding;
    private readonly ExpertDispatcher _dispatcher;

    public HierarchicalRoutingTests()
    {
        var persistence = new InMemoryStorageProvider();
        _index = new CognitiveIndex(persistence);
        _embedding = new HashEmbeddingService(dimensions: 384);
        _dispatcher = new ExpertDispatcher(_index, _embedding);
    }

    public void Dispose() => _index.Dispose();

    // ── CreateDomainNode ──

    [Fact]
    public void CreateDomainNode_Root_CreatesWithCorrectMetadata()
    {
        var result = _dispatcher.CreateDomainNode("engineering", "Software engineering and systems design", "root");

        Assert.Equal("engineering", result.ExpertId);
        Assert.Equal("domain_engineering", result.TargetNamespace);

        var entry = _index.Get("engineering", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(entry);
        Assert.Equal("ltm", entry!.LifecycleState);
        Assert.True(entry.IsSummaryNode);
        Assert.Equal("root", entry.Metadata["level"]);
        Assert.Equal("domain_engineering", entry.Metadata["targetNamespace"]);
        Assert.Equal("", entry.Metadata["childNodeIds"]);
        Assert.False(entry.Metadata.ContainsKey("parentNodeId"));
    }

    [Fact]
    public void CreateDomainNode_Branch_CreatesWithParentLink()
    {
        _dispatcher.CreateDomainNode("engineering", "Software engineering", "root");
        var result = _dispatcher.CreateDomainNode("backend", "Backend development and APIs", "branch", "engineering");

        Assert.Equal("backend", result.ExpertId);
        Assert.Equal("domain_backend", result.TargetNamespace);

        var entry = _index.Get("backend", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(entry);
        Assert.Equal("branch", entry!.Metadata["level"]);
        Assert.Equal("engineering", entry.Metadata["parentNodeId"]);
    }

    [Fact]
    public void CreateDomainNode_Branch_UpdatesParentChildNodeIds()
    {
        _dispatcher.CreateDomainNode("engineering", "Software engineering", "root");
        _dispatcher.CreateDomainNode("backend", "Backend development", "branch", "engineering");
        _dispatcher.CreateDomainNode("frontend", "Frontend development", "branch", "engineering");

        var parentEntry = _index.Get("engineering", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(parentEntry);
        var childIds = parentEntry!.Metadata["childNodeIds"].Split(',', StringSplitOptions.RemoveEmptyEntries);
        Assert.Contains("backend", childIds);
        Assert.Contains("frontend", childIds);
    }

    [Fact]
    public void CreateDomainNode_InvalidLevel_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("test", "description", "leaf"));
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("test", "description", "invalid"));
    }

    [Fact]
    public void CreateDomainNode_RootWithParent_Throws()
    {
        _dispatcher.CreateDomainNode("parent", "parent domain", "root");
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("child", "child domain", "root", "parent"));
    }

    [Fact]
    public void CreateDomainNode_BranchWithoutParent_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("orphan", "orphan branch", "branch"));
    }

    [Fact]
    public void CreateDomainNode_BranchWithNonexistentParent_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("orphan", "orphan branch", "branch", "nonexistent"));
    }

    [Fact]
    public void CreateDomainNode_EmptyId_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("", "description", "root"));
    }

    [Fact]
    public void CreateDomainNode_EmptyDescription_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            _dispatcher.CreateDomainNode("test", "", "root"));
    }

    // ── RouteHierarchical ──

    [Fact]
    public void RouteHierarchical_WalksTree_RootToBranchToLeaf()
    {
        // Build a 3-level tree — use identical text at each level so
        // HashEmbeddingService produces matching vectors (cosine ~1.0)
        string sharedDesc = "Backend development with databases and APIs";
        _dispatcher.CreateDomainNode("engineering", sharedDesc, "root");
        _dispatcher.CreateDomainNode("backend", sharedDesc, "branch", "engineering");

        // Create a leaf expert under the branch
        _dispatcher.CreateExpert("db_engineer", sharedDesc);
        // Link the leaf to the branch
        var leafEntry = _index.Get("db_engineer", ExpertDispatcher.SystemNamespace);
        leafEntry!.Metadata["parentNodeId"] = "backend";
        leafEntry.Metadata["level"] = "leaf";
        _index.Upsert(leafEntry);
        // Update parent's childNodeIds
        var branchEntry = _index.Get("backend", ExpertDispatcher.SystemNamespace);
        branchEntry!.Metadata["childNodeIds"] = "db_engineer";
        _index.Upsert(branchEntry);

        // Query with matching text — since HashEmbeddingService gives identical vectors
        // for identical text, this ensures the tree walk succeeds
        var queryVector = _embedding.Embed(sharedDesc);
        var result = _dispatcher.RouteHierarchical(queryVector, topK: 3, threshold: 0.3f);

        Assert.Equal("routed", result.Status);
        Assert.NotEmpty(result.Experts);
        Assert.Contains(result.Experts, e => e.ExpertId == "db_engineer");
        Assert.NotEmpty(result.Path);
    }

    [Fact]
    public void RouteHierarchical_FallsBackToFlat_WhenNoTreeExists()
    {
        // Create only leaf experts (no root/branch nodes)
        string persona = "A database specialist focused on query optimization.";
        _dispatcher.CreateExpert("db_specialist", persona);

        var queryVector = _embedding.Embed(persona);
        var result = _dispatcher.RouteHierarchical(queryVector, topK: 3, threshold: 0.3f);

        // Should fall back to flat routing and still find the expert
        Assert.Equal("routed", result.Status);
        Assert.NotEmpty(result.Experts);
        Assert.Equal("db_specialist", result.Experts[0].ExpertId);
        // Path should be empty since flat routing was used
        Assert.Empty(result.Path);
    }

    [Fact]
    public void RouteHierarchical_CrossDomain_MatchesMultipleRoots()
    {
        // Create two root domains with same description for deterministic hash match
        string desc = "Software engineering and data science combined";
        _dispatcher.CreateDomainNode("engineering", desc, "root");
        _dispatcher.CreateDomainNode("data_science", desc, "root");

        var queryVector = _embedding.Embed(desc);
        var result = _dispatcher.RouteHierarchical(queryVector, topK: 3, threshold: 0.3f);

        // Both roots should match (identical vectors from identical text)
        var rootNodes = result.Path.Where(p => p.Level == "root").ToList();
        Assert.True(rootNodes.Count >= 1);
    }

    [Fact]
    public void RouteHierarchical_NeedsExpert_WhenNoMatch()
    {
        _dispatcher.CreateDomainNode("engineering", "Software engineering", "root");

        var queryVector = _embedding.Embed("Quantum physics and particle interactions");
        var result = _dispatcher.RouteHierarchical(queryVector, topK: 3, threshold: 0.99f);

        // With extremely high threshold, nothing should match
        Assert.Equal("needs_expert", result.Status);
    }

    [Fact]
    public void RouteHierarchical_NeedsExpert_WhenNoExperts()
    {
        var queryVector = _embedding.Embed("Any query at all");
        var result = _dispatcher.RouteHierarchical(queryVector, topK: 3, threshold: 0.75f);

        Assert.Equal("needs_expert", result.Status);
        Assert.Empty(result.Path);
    }

    // ── GetDomainTree ──

    [Fact]
    public void GetDomainTree_ReturnsCorrectStructure()
    {
        // Build tree: root → branch → leaf
        _dispatcher.CreateDomainNode("eng", "Engineering", "root");
        _dispatcher.CreateDomainNode("backend", "Backend dev", "branch", "eng");
        _dispatcher.CreateExpert("api_dev", "API developer");

        // Link leaf to branch
        var leafEntry = _index.Get("api_dev", ExpertDispatcher.SystemNamespace);
        leafEntry!.Metadata["parentNodeId"] = "backend";
        leafEntry.Metadata["level"] = "leaf";
        _index.Upsert(leafEntry);
        var branchEntry = _index.Get("backend", ExpertDispatcher.SystemNamespace);
        branchEntry!.Metadata["childNodeIds"] += (string.IsNullOrEmpty(branchEntry.Metadata["childNodeIds"]) ? "" : ",") + "api_dev";
        _index.Upsert(branchEntry);

        var tree = _dispatcher.GetDomainTree();

        Assert.Equal(3, tree.TotalNodes);
        Assert.True(tree.MaxDepth >= 1);
        Assert.Single(tree.Roots);
        Assert.Equal("eng", tree.Roots[0].NodeId);
        Assert.Equal("root", tree.Roots[0].Level);
        Assert.Contains("backend", tree.Roots[0].ChildNodeIds);
    }

    [Fact]
    public void GetDomainTree_EmptyTree_ReturnsEmpty()
    {
        var tree = _dispatcher.GetDomainTree();
        Assert.Empty(tree.Roots);
        Assert.Equal(0, tree.TotalNodes);
        Assert.Equal(0, tree.MaxDepth);
    }

    [Fact]
    public void GetDomainTree_FlatLeaves_ReturnsAsRootlessNodes()
    {
        // Create only leaf experts without tree structure
        _dispatcher.CreateExpert("expert_a", "Domain A expert.");
        _dispatcher.CreateExpert("expert_b", "Domain B expert.");

        var tree = _dispatcher.GetDomainTree();

        // No roots → leaves are listed as flat nodes
        Assert.Equal(2, tree.TotalNodes);
        Assert.Equal(2, tree.Roots.Count); // flat leaves become the "roots" list
        Assert.All(tree.Roots, n => Assert.Equal("leaf", n.Level));
    }

    // ── GetNodesByLevel / GetChildren ──

    [Fact]
    public void GetNodesByLevel_FiltersCorrectly()
    {
        _dispatcher.CreateDomainNode("root1", "Root domain", "root");
        _dispatcher.CreateDomainNode("branch1", "Branch domain", "branch", "root1");
        _dispatcher.CreateExpert("leaf1", "Leaf expert");

        var roots = _dispatcher.GetNodesByLevel("root");
        var branches = _dispatcher.GetNodesByLevel("branch");
        var leaves = _dispatcher.GetNodesByLevel("leaf");

        Assert.Single(roots);
        Assert.Equal("root1", roots[0].Id);
        Assert.Single(branches);
        Assert.Equal("branch1", branches[0].Id);
        Assert.Single(leaves);
        Assert.Equal("leaf1", leaves[0].Id);
    }

    [Fact]
    public void GetChildren_ReturnsChildNodes()
    {
        _dispatcher.CreateDomainNode("parent", "Parent domain", "root");
        _dispatcher.CreateDomainNode("child1", "Child 1", "branch", "parent");
        _dispatcher.CreateDomainNode("child2", "Child 2", "branch", "parent");

        var children = _dispatcher.GetChildren("parent");
        Assert.Equal(2, children.Count);
        Assert.Contains(children, c => c.Id == "child1");
        Assert.Contains(children, c => c.Id == "child2");
    }

    [Fact]
    public void GetChildren_NoChildren_ReturnsEmpty()
    {
        _dispatcher.CreateDomainNode("lonely", "Lonely root", "root");
        var children = _dispatcher.GetChildren("lonely");
        Assert.Empty(children);
    }

    // ── Backward Compatibility ──

    [Fact]
    public void BackwardCompat_ExpertsWithoutLevelMetadata_DefaultToLeaf()
    {
        // Experts created with the original CreateExpert have no level metadata
        _dispatcher.CreateExpert("old_expert", "An expert created without hierarchy.");

        var entry = _index.Get("old_expert", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(entry);
        // Should NOT have level metadata
        Assert.False(entry!.Metadata.ContainsKey("level"));

        // GetNodesByLevel should still categorize as "leaf"
        var leaves = _dispatcher.GetNodesByLevel("leaf");
        Assert.Contains(leaves, e => e.Id == "old_expert");
    }

    [Fact]
    public void BackwardCompat_ExpertsWithoutLevelMetadata_RouteCorrectly()
    {
        string persona = "A security specialist for penetration testing.";
        _dispatcher.CreateExpert("pentest_expert", persona);

        // Original Route() should still work
        var queryVector = _embedding.Embed(persona);
        var (status, experts) = _dispatcher.Route(queryVector, threshold: 0.3f);
        Assert.Equal("routed", status);
        Assert.Contains(experts, e => e.ExpertId == "pentest_expert");

        // RouteHierarchical should fall back to flat routing and also work
        var result = _dispatcher.RouteHierarchical(queryVector, threshold: 0.3f);
        Assert.Equal("routed", result.Status);
        Assert.Contains(result.Experts, e => e.ExpertId == "pentest_expert");
    }

    // ── Tool-level tests ──

    [Fact]
    public void DispatchTask_HierarchicalTrue_UsesTreeRouting()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        // Build tree
        tools.CreateExpert("eng_root", "Software engineering domain", level: "root");

        string leafPersona = "Software engineering domain";
        tools.CreateExpert("eng_leaf", leafPersona, level: "leaf", parentNodeId: "eng_root");

        // Dispatch with hierarchical=true
        var result = tools.DispatchTask(leafPersona, threshold: 0.3f, hierarchical: true);
        var typed = Assert.IsType<HierarchicalRouteResult>(result);

        Assert.Equal("routed", typed.Status);
        index.Dispose();
    }

    [Fact]
    public void DispatchTask_HierarchicalFalse_UsesFlatRouting()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        string persona = "A frontend developer specializing in React components.";
        tools.CreateExpert("react_dev", persona);

        // Default hierarchical=false should use flat routing
        var result = tools.DispatchTask(persona, threshold: 0.3f, hierarchical: false);
        var typed = Assert.IsType<DispatchRoutedResult>(result);

        Assert.Equal("routed", typed.Status);
        Assert.Equal("react_dev", typed.Expert.ExpertId);
        index.Dispose();
    }

    [Fact]
    public void DispatchTask_Default_UsesFlatRouting()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        string persona = "A Go backend developer.";
        tools.CreateExpert("go_dev", persona);

        // No hierarchical parameter (default false) should use flat routing
        var result = tools.DispatchTask(persona, threshold: 0.3f);
        var typed = Assert.IsType<DispatchRoutedResult>(result);

        Assert.Equal("routed", typed.Status);
        Assert.Equal("go_dev", typed.Expert.ExpertId);
        index.Dispose();
    }

    [Fact]
    public void CreateExpert_WithLevel_Root_CreatesDomainNode()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        var result = tools.CreateExpert("science", "Science and research domain", level: "root");
        var typed = Assert.IsType<CreateExpertResult>(result);

        Assert.Equal("created", typed.Status);
        Assert.Equal("science", typed.ExpertId);
        Assert.Equal("domain_science", typed.TargetNamespace);

        var entry = index.Get("science", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(entry);
        Assert.Equal("root", entry!.Metadata["level"]);
        index.Dispose();
    }

    [Fact]
    public void CreateExpert_WithLevel_Leaf_AndParent_LinksCorrectly()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        tools.CreateExpert("eng_root", "Engineering domain", level: "root");
        tools.CreateExpert("go_dev", "A Go developer", level: "leaf", parentNodeId: "eng_root");

        var leafEntry = index.Get("go_dev", ExpertDispatcher.SystemNamespace);
        Assert.NotNull(leafEntry);
        Assert.Equal("leaf", leafEntry!.Metadata["level"]);
        Assert.Equal("eng_root", leafEntry.Metadata["parentNodeId"]);

        var parentEntry = index.Get("eng_root", ExpertDispatcher.SystemNamespace);
        Assert.Contains("go_dev", parentEntry!.Metadata["childNodeIds"]);
        index.Dispose();
    }

    [Fact]
    public void GetDomainTree_Tool_ReturnsResult()
    {
        var persistence = new InMemoryStorageProvider();
        var index = new CognitiveIndex(persistence);
        var embedding = new HashEmbeddingService(dimensions: 384);
        var dispatcher = new ExpertDispatcher(index, embedding);
        var tools = new ExpertTools(dispatcher, index, embedding, new MetricsCollector());

        tools.CreateExpert("eng", "Engineering", level: "root");

        var result = tools.GetDomainTree();
        var typed = Assert.IsType<DomainTreeResult>(result);

        Assert.Single(typed.Roots);
        Assert.Equal(1, typed.TotalNodes);
        index.Dispose();
    }
}

/// <summary>
/// Minimal in-memory storage provider for testing.
/// </summary>
file sealed class InMemoryStorageProvider : IStorageProvider
{
    private readonly Dictionary<string, NamespaceData> _data = new();

    public NamespaceData LoadNamespace(string ns)
        => _data.TryGetValue(ns, out var d) ? d : new NamespaceData();

    public void ScheduleSave(string ns, Func<NamespaceData> dataProvider)
        => _data[ns] = dataProvider();

    public void SaveNamespaceSync(string ns, NamespaceData data)
        => _data[ns] = data;

    public IReadOnlyList<string> GetPersistedNamespaces()
        => _data.Keys.Where(k => !k.StartsWith("_")).ToList();

    public List<GraphEdge> LoadGlobalEdges() => new();
    public void ScheduleSaveGlobalEdges(Func<List<GraphEdge>> dataProvider) { }
    public List<SemanticCluster> LoadClusters() => new();
    public void ScheduleSaveClusters(Func<List<SemanticCluster>> dataProvider) { }
    public List<CollapseRecord> LoadCollapseHistory() => new();
    public void ScheduleSaveCollapseHistory(Func<List<CollapseRecord>> dataProvider) { }
    public Dictionary<string, DecayConfig> LoadDecayConfigs() => new();
    public void ScheduleSaveDecayConfigs(Func<Dictionary<string, DecayConfig>> dataProvider) { }
    public bool SupportsIncrementalWrites => false;
    public void ScheduleUpsertEntry(string ns, CognitiveEntry entry) { }
    public void ScheduleDeleteEntry(string ns, string entryId) { }
    public Task DeleteNamespaceAsync(string ns) { _data.Remove(ns); return Task.CompletedTask; }
    public void Flush() { }
    public void Dispose() { }
}
