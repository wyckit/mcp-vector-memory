using System.Text.Json.Serialization;

namespace McpEngramMemory.Core.Models;

/// <summary>
/// A matched expert from the semantic routing meta-index.
/// </summary>
public sealed record ExpertMatch(
    [property: JsonPropertyName("expertId")] string ExpertId,
    [property: JsonPropertyName("personaDescription")] string PersonaDescription,
    [property: JsonPropertyName("targetNamespace")] string TargetNamespace,
    [property: JsonPropertyName("score")] float Score,
    [property: JsonPropertyName("taskCount")] int TaskCount);

/// <summary>
/// Result of dispatch_task when an expert is found (status = "routed").
/// </summary>
public sealed record DispatchRoutedResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("expert")] ExpertMatch Expert,
    [property: JsonPropertyName("candidateExperts")] IReadOnlyList<ExpertMatch> CandidateExperts,
    [property: JsonPropertyName("context")] IReadOnlyList<CognitiveSearchResult> Context);

/// <summary>
/// Result of dispatch_task when no expert matches (status = "needs_expert").
/// </summary>
public sealed record DispatchMissResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("closestExperts")] IReadOnlyList<ExpertMatch> ClosestExperts,
    [property: JsonPropertyName("suggestion")] string Suggestion);

/// <summary>
/// Result of create_expert tool. Includes optional placement info from auto-classification.
/// </summary>
public sealed record CreateExpertResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("expertId")] string ExpertId,
    [property: JsonPropertyName("targetNamespace")] string TargetNamespace,
    [property: JsonPropertyName("placement")] PlacementInfo? Placement = null);

/// <summary>
/// Auto-classification placement result from the domain tree.
/// Status is "auto_linked" (>= 0.82), "suggested" (0.60–0.82), or "unclassified" (&lt; 0.60).
/// </summary>
public sealed record PlacementInfo(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("parentNodeId")] string? ParentNodeId,
    [property: JsonPropertyName("confidence")] float Confidence,
    [property: JsonPropertyName("candidates")] IReadOnlyList<PlacementCandidate> Candidates);

/// <summary>
/// A candidate parent node for expert placement.
/// </summary>
public sealed record PlacementCandidate(
    [property: JsonPropertyName("nodeId")] string NodeId,
    [property: JsonPropertyName("level")] string Level,
    [property: JsonPropertyName("description")] string Description,
    [property: JsonPropertyName("score")] float Score);

/// <summary>
/// A node in the hierarchical domain tree (root, branch, or leaf).
/// </summary>
public sealed record DomainNode(
    [property: JsonPropertyName("nodeId")] string NodeId,
    [property: JsonPropertyName("description")] string Description,
    [property: JsonPropertyName("targetNamespace")] string TargetNamespace,
    [property: JsonPropertyName("level")] string Level,
    [property: JsonPropertyName("parentNodeId")] string? ParentNodeId,
    [property: JsonPropertyName("childNodeIds")] IReadOnlyList<string> ChildNodeIds,
    [property: JsonPropertyName("score")] float Score,
    [property: JsonPropertyName("taskCount")] int TaskCount);

/// <summary>
/// Result of hierarchical routing through the domain tree (root → branch → leaf).
/// </summary>
public sealed record HierarchicalRouteResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("path")] IReadOnlyList<DomainNode> Path,
    [property: JsonPropertyName("experts")] IReadOnlyList<ExpertMatch> Experts,
    [property: JsonPropertyName("context")] IReadOnlyList<CognitiveSearchResult> Context);

/// <summary>
/// Full domain tree structure showing roots, branches, and leaves.
/// </summary>
public sealed record DomainTreeResult(
    [property: JsonPropertyName("roots")] IReadOnlyList<DomainNode> Roots,
    [property: JsonPropertyName("totalNodes")] int TotalNodes,
    [property: JsonPropertyName("maxDepth")] int MaxDepth);
