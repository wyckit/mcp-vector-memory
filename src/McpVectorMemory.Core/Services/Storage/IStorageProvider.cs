using McpVectorMemory.Core.Models;

namespace McpVectorMemory.Core.Services.Storage;

/// <summary>
/// Abstraction for data persistence. Implementations handle loading, saving,
/// and debounced writes for namespace data, graph edges, and clusters.
/// </summary>
public interface IStorageProvider : IDisposable
{
    NamespaceData LoadNamespace(string ns);
    IReadOnlyList<string> GetPersistedNamespaces();
    void ScheduleSave(string ns, Func<NamespaceData> dataProvider);
    void SaveNamespaceSync(string ns, NamespaceData data);

    List<GraphEdge> LoadGlobalEdges();
    void ScheduleSaveGlobalEdges(Func<List<GraphEdge>> dataProvider);

    List<SemanticCluster> LoadClusters();
    void ScheduleSaveClusters(Func<List<SemanticCluster>> dataProvider);

    List<CollapseRecord> LoadCollapseHistory();
    void ScheduleSaveCollapseHistory(Func<List<CollapseRecord>> dataProvider);

    Dictionary<string, DecayConfig> LoadDecayConfigs();
    void ScheduleSaveDecayConfigs(Func<Dictionary<string, DecayConfig>> dataProvider);

    void Flush();
}
