using McpVectorMemory.Models;

namespace McpVectorMemory.Services;

/// <summary>
/// Manages activation energy computation, decay cycles, and lifecycle state transitions.
/// </summary>
public sealed class LifecycleEngine
{
    private readonly CognitiveIndex _index;

    public LifecycleEngine(CognitiveIndex index)
    {
        _index = index;
    }

    /// <summary>
    /// Trigger activation energy recomputation and state transitions.
    /// Formula: ActivationEnergy = (accessCount * reinforcementWeight) - (hoursSinceLastAccess * decayRate)
    /// </summary>
    public DecayCycleResult RunDecayCycle(
        string ns,
        float decayRate = 0.1f,
        float reinforcementWeight = 1.0f,
        float stmThreshold = 2.0f,
        float archiveThreshold = -5.0f)
    {
        var allNamespaces = ns == "*" ? _index.GetNamespaces() : new[] { ns };

        var stmToLtmIds = new List<string>();
        var ltmToArchivedIds = new List<string>();
        int processedCount = 0;

        foreach (var currentNs in allNamespaces)
        {
            // GetAllInNamespace returns a snapshot list — safe to iterate
            var entries = _index.GetAllInNamespace(currentNs);
            foreach (var entry in entries)
            {
                if (entry.IsSummaryNode) continue; // Don't decay summary nodes

                processedCount++;
                var hoursSinceAccess = (float)(DateTimeOffset.UtcNow - entry.LastAccessedAt).TotalHours;
                float newActivationEnergy = (entry.AccessCount * reinforcementWeight) - (hoursSinceAccess * decayRate);

                // Determine new state
                string? newState = null;
                switch (entry.LifecycleState)
                {
                    case "stm" when newActivationEnergy < stmThreshold:
                        newState = "ltm";
                        stmToLtmIds.Add(entry.Id);
                        break;
                    case "ltm" when newActivationEnergy < archiveThreshold:
                        newState = "archived";
                        ltmToArchivedIds.Add(entry.Id);
                        break;
                }

                // Atomically update activation energy and optional state transition via the index
                _index.SetActivationEnergyAndState(entry.Id, newActivationEnergy, newState);
            }
        }

        return new DecayCycleResult(
            processedCount,
            stmToLtmIds.Count,
            ltmToArchivedIds.Count,
            stmToLtmIds,
            ltmToArchivedIds);
    }

    /// <summary>Promote (or demote) an entry to a specific lifecycle state.</summary>
    public string PromoteMemory(string id, string targetState)
    {
        if (targetState is not ("stm" or "ltm" or "archived"))
            return $"Error: Invalid target state '{targetState}'. Use 'stm', 'ltm', or 'archived'.";

        var entry = _index.Get(id);
        if (entry is null)
            return $"Error: Entry '{id}' not found.";

        var previousState = entry.LifecycleState;
        if (!_index.SetLifecycleState(id, targetState))
            return $"Error: Failed to update state for '{id}'.";

        return $"Entry '{id}' transitioned: {previousState} -> {targetState}.";
    }

    /// <summary>Deep recall: search all states and auto-resurrect high-scoring archived entries.</summary>
    public IReadOnlyList<CognitiveSearchResult> DeepRecall(
        float[] vector, string ns, int k = 10, float minScore = 0.3f,
        float resurrectionThreshold = 0.7f)
    {
        var results = _index.SearchAllStates(vector, ns, k, minScore);

        // Auto-resurrect high-scoring archived entries and return updated results
        var updatedResults = new List<CognitiveSearchResult>(results.Count);
        foreach (var result in results)
        {
            if (result.LifecycleState == "archived" && result.Score >= resurrectionThreshold)
            {
                _index.SetLifecycleState(result.Id, "stm");
                _index.RecordAccess(result.Id);
                // Return with updated lifecycle state
                updatedResults.Add(result with { LifecycleState = "stm" });
            }
            else
            {
                updatedResults.Add(result);
            }
        }

        return updatedResults;
    }
}
