using McpEngramMemory.Core.Models;
using McpEngramMemory.Core.Services.Retrieval;

namespace McpEngramMemory.Core.Services.Intelligence;

/// <summary>
/// Generates extractive summaries from clusters of related memory entries
/// without requiring an LLM. Uses keyword extraction and representative entry
/// selection to produce searchable summary text.
/// </summary>
public static class AutoSummarizer
{
    private const int MaxKeywords = 8;
    private const int MaxRepresentativeSnippets = 3;
    private const int MaxSnippetLength = 120;

    /// <summary>
    /// Generate a summary from cluster members using extractive keyword analysis.
    /// Returns the summary text and the centroid vector for embedding.
    /// </summary>
    public static string GenerateSummary(IReadOnlyList<CognitiveEntry> members)
    {
        if (members.Count == 0)
            return "Empty cluster.";

        // Extract keywords by term frequency across all member texts
        var keywords = ExtractKeywords(members);

        // Find the most representative entry (closest to centroid conceptually — use longest text as proxy)
        var snippets = GetRepresentativeSnippets(members);

        // Build structured summary text
        var keywordStr = keywords.Count > 0
            ? string.Join(", ", keywords)
            : "general";

        var parts = new List<string>
        {
            $"Cluster of {members.Count} related memories about: {keywordStr}."
        };

        // Add category breakdown if mixed
        var categories = members
            .Where(m => !string.IsNullOrWhiteSpace(m.Category))
            .GroupBy(m => m.Category!)
            .OrderByDescending(g => g.Count())
            .Take(3)
            .Select(g => $"{g.Key} ({g.Count()})")
            .ToList();

        if (categories.Count > 0)
            parts.Add($"Categories: {string.Join(", ", categories)}.");

        // Add representative snippets
        if (snippets.Count > 0)
        {
            parts.Add("Key entries:");
            foreach (var snippet in snippets)
                parts.Add($"- {snippet}");
        }

        return string.Join(" ", parts);
    }

    /// <summary>
    /// Extract top keywords from member texts using term frequency.
    /// Reuses BM25Index.Tokenize for consistent tokenization.
    /// </summary>
    public static List<string> ExtractKeywords(IReadOnlyList<CognitiveEntry> members)
    {
        var termFreq = new Dictionary<string, int>();
        var docFreq = new Dictionary<string, int>();

        foreach (var member in members)
        {
            if (string.IsNullOrWhiteSpace(member.Text)) continue;

            var tokens = BM25Index.Tokenize(member.Text);
            var uniqueTokens = new HashSet<string>(tokens);

            foreach (var token in tokens)
                termFreq[token] = termFreq.GetValueOrDefault(token) + 1;

            foreach (var token in uniqueTokens)
                docFreq[token] = docFreq.GetValueOrDefault(token) + 1;
        }

        // Score by TF * IDF-like factor: terms frequent across many docs are more representative
        // But penalize terms that appear in ALL documents (too generic)
        int docCount = members.Count(m => !string.IsNullOrWhiteSpace(m.Text));
        if (docCount == 0) return new();

        return termFreq
            .Where(kv => kv.Value >= 2) // Must appear at least twice
            .Where(kv => kv.Key.Length >= 3) // Skip very short tokens
            .Select(kv =>
            {
                int df = docFreq.GetValueOrDefault(kv.Key);
                // Favor terms in multiple docs but not all — IDF-like
                float idf = df < docCount ? MathF.Log((float)docCount / df + 1f) : 0.5f;
                float score = kv.Value * idf;
                return (term: kv.Key, score);
            })
            .OrderByDescending(x => x.score)
            .Take(MaxKeywords)
            .Select(x => x.term)
            .ToList();
    }

    /// <summary>
    /// Get representative text snippets from the most informative entries.
    /// </summary>
    private static List<string> GetRepresentativeSnippets(IReadOnlyList<CognitiveEntry> members)
    {
        return members
            .Where(m => !string.IsNullOrWhiteSpace(m.Text))
            .OrderByDescending(m => m.Text!.Length) // Longer text = more informative
            .ThenByDescending(m => m.AccessCount) // Break ties by usage
            .Take(MaxRepresentativeSnippets)
            .Select(m => Truncate(m.Text!, MaxSnippetLength))
            .ToList();
    }

    private static string Truncate(string text, int maxLength)
    {
        if (text.Length <= maxLength) return text;

        // Try to cut at a word boundary
        int cutAt = text.LastIndexOf(' ', maxLength - 3);
        if (cutAt < maxLength / 2) cutAt = maxLength - 3;
        return text[..cutAt] + "...";
    }
}
