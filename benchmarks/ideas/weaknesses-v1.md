# Default-v1 Benchmark Weaknesses

Analysis of the 7 queries where Recall@5 < 1.0 in the current benchmark run (2026-03-07).

## Weak Queries

### q02 — "Systems programming with memory safety" (Recall: 0.67)
- **Retrieved:** bench-rust, bench-memmanage, bench-gc, bench-concurrency, bench-python
- **Missing:** bench-csharp (grade 1)
- **Analysis:** "memory safety" pulled in memory management topics; C# is only marginally relevant (grade 1) so low impact.

### q03 — "Fast key-value data storage" (Recall: 0.67)
- **Retrieved:** bench-hashtable, bench-memmanage, bench-gc, bench-rust, bench-sql
- **Missing:** bench-nosql (grade 2)
- **Analysis:** "storage" pulled in memory topics instead of NoSQL. The embedding doesn't strongly associate "key-value" with NoSQL document stores.

### q07 — "Network communication protocols" (Recall: 0.67)
- **Retrieved:** bench-tcpip, bench-http, bench-neuralnet, bench-graph, bench-concurrency
- **Missing:** bench-restapi (grade 1)
- **Analysis:** "neural net" and "graph" ranked above REST API. The word "network" has dual meaning (neural network vs computer network).

### q12 — "Semantic search and embeddings" (Recall: 0.67)
- **Retrieved:** bench-embeddings, bench-vectorsearch, bench-binarytree, bench-graph, bench-sql
- **Missing:** bench-cosine (grade 2)
- **Analysis:** "search" pulled in tree/graph data structures. Cosine similarity is semantically related but doesn't share surface terms with "semantic search."

### q15 — "Linked vs array-based data structures" (Recall: 0.50)
- **Retrieved:** bench-linkedlist, bench-hashtable, bench-sql, bench-binarytree, bench-graph
- **Missing:** bench-sorting (grade 1)
- **Analysis:** Sorting is only marginally relevant (grade 1). The query is specific to linked lists and arrays; sorting is tangential.

### q16 — "Graph traversal and network analysis" (Recall: 0.50)
- **Retrieved:** bench-graph, bench-neuralnet, bench-embeddings, bench-vectorsearch, bench-transformer
- **Missing:** bench-binarytree (grade 1)
- **Analysis:** "network" again pulls in neural network topics. The embedding model conflates "network analysis" with neural networks.

### q17 — "Attention mechanisms in NLP" (Recall: 0.67)
- **Retrieved:** bench-transformer, bench-cosine, bench-neuralnet, bench-mutex, bench-tcpip
- **Missing:** bench-embeddings (grade 1)
- **Analysis:** "attention" pulled in mutex (attention/synchronization?) and TCP. Embeddings are only marginally relevant (grade 1).

## Patterns

1. **"Network" ambiguity** — Queries q07, q16 both suffer from "network" matching neural networks instead of computer networks.
2. **"Memory/storage" ambiguity** — Query q03 conflates key-value storage with memory management.
3. **Low-grade misses** — Most missing entries are grade 1 (marginally relevant). The system reliably retrieves grade 2-3 entries.
4. **MRR is perfect** — The most relevant result is always #1, so the ranking head is strong even when the tail is weak.
