using McpVectorMemory.Core.Models;
using McpVectorMemory.Core.Services;
using McpVectorMemory.Core.Services.Evaluation;
using McpVectorMemory.Core.Services.Experts;
using McpVectorMemory.Core.Services.Graph;
using McpVectorMemory.Core.Services.Intelligence;
using McpVectorMemory.Core.Services.Lifecycle;
using McpVectorMemory.Core.Services.Retrieval;
using McpVectorMemory.Core.Services.Storage;
using McpVectorMemory.Tools;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

var builder = Host.CreateApplicationBuilder(args);

builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);

// Storage provider — set MEMORY_STORAGE=sqlite to use SQLite backend
var storageMode = Environment.GetEnvironmentVariable("MEMORY_STORAGE");
if (string.Equals(storageMode, "sqlite", StringComparison.OrdinalIgnoreCase))
{
    var dbPath = Environment.GetEnvironmentVariable("MEMORY_SQLITE_PATH");
    builder.Services.AddSingleton(sp => new SqliteStorageProvider(
        dbPath: dbPath, logger: sp.GetService<ILogger<SqliteStorageProvider>>()));
    builder.Services.AddSingleton<IStorageProvider>(sp => sp.GetRequiredService<SqliteStorageProvider>());
}
else
{
    builder.Services.AddSingleton(sp => new PersistenceManager(
        logger: sp.GetService<ILogger<PersistenceManager>>()));
    builder.Services.AddSingleton<IStorageProvider>(sp => sp.GetRequiredService<PersistenceManager>());
}
// Memory limits — configurable via environment variables
var limits = new MemoryLimitsConfig(
    MaxNamespaceSize: int.TryParse(Environment.GetEnvironmentVariable("MEMORY_MAX_NAMESPACE_SIZE"), out var maxNs) ? maxNs : int.MaxValue,
    MaxTotalCount: int.TryParse(Environment.GetEnvironmentVariable("MEMORY_MAX_TOTAL_COUNT"), out var maxTotal) ? maxTotal : int.MaxValue);
builder.Services.AddSingleton(limits);
builder.Services.AddSingleton<CognitiveIndex>();
builder.Services.AddSingleton<KnowledgeGraph>();
builder.Services.AddSingleton<ClusterManager>();
builder.Services.AddSingleton<LifecycleEngine>();
builder.Services.AddSingleton<PhysicsEngine>();
builder.Services.AddSingleton<AccretionScanner>();
builder.Services.AddSingleton<MetricsCollector>();
builder.Services.AddSingleton<QueryExpander>();
builder.Services.AddSingleton<BenchmarkRunner>();
builder.Services.AddSingleton<DebateSessionManager>();
builder.Services.AddSingleton<ExpertDispatcher>();
builder.Services.AddSingleton<OnnxEmbeddingService>();
builder.Services.AddSingleton<IEmbeddingService>(sp => sp.GetRequiredService<OnnxEmbeddingService>());
builder.Services.AddHostedService<EmbeddingWarmupService>();
builder.Services.AddHostedService<DecayBackgroundService>();
builder.Services.AddHostedService<AccretionBackgroundService>();

// MCP Server with all tool groups
builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithTools<CoreMemoryTools>()
    .WithTools<GraphTools>()
    .WithTools<ClusterTools>()
    .WithTools<LifecycleTools>()
    .WithTools<AdminTools>()
    .WithTools<AccretionTools>()
    .WithTools<BenchmarkTools>()
    .WithTools<IntelligenceTools>()
    .WithTools<DebateTools>()
    .WithTools<MaintenanceTools>()
    .WithTools<ExpertTools>();

await builder.Build().RunAsync();
