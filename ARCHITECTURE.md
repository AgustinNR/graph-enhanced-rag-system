# Architecture Decision Record: Graph-Enhanced RAG System

## ADR-001: Graph-Enhanced RAG System Architecture

**Status:** Proposed  
**Date:** 2025-09-01  
**Authors:** Agustin Rodriguez  
**Reviewers:** [Team Lead, Senior Engineer]

---

## Context

The solution proposes an AI support assistant that combines Neo4j knowledge graphs with embeddings and LLM integration to answer technical support questions. The system must provide fast responses (millisecond to one second range) even when dealing with potentially billions of nodes and edges in the knowledge graph.

### Requirements
- Sub-second response times for common queries
- Semantic search capability using embeddings
- Multi-hop graph traversal for context enrichment
- Scalability to billion-node graphs

### Constraints
- Neo4j 5.x infrastructure
- Python runtime environment

---

## Decision Categories

### 1. Embedding Strategy & Storage

### 1.1 Embedding Strategy

**Decision:** Use the Sentence-Transformers `all-MiniLM-L6-v2` model (384-dimensional embeddings) as primary embedding model for both data indexing and user queries.

**Context:** Need to balance embedding model performance, latency, and storage efficiency.

**Options Considered:**
- [x] Option A: sentence-transformers/all-MiniLM-L6-v2 (384d)
- [ ] Option B: sentence-transformers/all-MiniLM-L12-v2 (384d)
- [ ] Option C: bge-small-en-v1.5 (384d)
- [ ] Option D: intfloat/e5-small-v2 (384d)
- [ ] Option E: OpenAI text-embedding-3-small (1536d) or text-embedding-3-large (3072d)


**Decision Rationale:**

* **Fit for purpose & Dimension Alignment (384-d):**
Stakeholders operate in English and the domain (smart-home support) involves concise queries and short passages. `all-MiniLM-L6-v2` is strong on retrieval tasks for this use case, with widely reported solid performance relative to its size. The model produces fixed-size vectors with 384 dimensions, which matches the Neo4j vector index configuration and avoids dimensionality mismatches during query time. 
* **Latency, Cost & Performance:**
Generating embeddings with MiniLM is fast (<10ms per short query on modern hardware). Lower dimensionality reduces vector storage size and speeds up similarity search (cosine distance complexity is O(d), so d=384 is ~4× faster than d=1536). Local inference avoids network/API latency and external costs.
* **Resource Efficiency, Storage & Scalability**
A 384-d float32 embedding requires ~1.5 KB. For millions of nodes, this significantly reduces memory and disk usage compared to higher-dimensional models (e.g., 1536-d → ~6 KB). This choice makes the system more scalable and cost-efficient.
* **Quality vs. Cost Trade-off:**
MiniLM embeddings provide sufficient semantic quality for technical support domains (short queries, FAQs, troubleshooting steps), where intent matching and entity recall are more important than nuanced language understanding.
Higher-dimensional models (768/1024/1536) might improve semantic recall but at the cost of increased storage, latency, and infrastructure overhead.
* **Open Source & Local Execution:**
Sentence-Transformers can be run locally without external API dependencies. This supports offline development, avoids vendor lock-in, and reduces operational costs (no per-request fees).

* **Alternatives considered:**
    - `all-MiniLM-L12-v2 (384d)`: Slightly larger encoder; marginal quality gains at higher compute/latency—benefit unlikely to justify cost for the workload.
    - `bge-small-en-v1.5 (384d)` / `e5-small-v2 (384d)`: Competitive on public leaderboards; viable drop-ins if later evaluation shows a measurable uplift. Can be swapped with minimal code/config changes because dimensions stay at 384.
    - `OpenAI (1536d/3072d)`: Strong quality but increases vector size 4–8×, introduces external dependency, variable latency, and ongoing cost.

### 1.2 Storage Strategy

**Decision:** Use Neo4j native vector indexes (5.x) to store/retrieve 384-d embeddings on `Product`, `Issue`, and `Solution` nodes with similarity_function: 'cosine'.  

**Options Considered:**
- [x] Neo4j vector indexes (native support in 5.x)
- [ ] TigerGraph
- [ ] Amazon Neptune
- [ ] ArangoDB

**Decision Rationale:**

- Local-first & simple: one database (graph + vectors), one query planner, one Docker compose.
- Good enough latency/recall for 384-d small/medium corpora; HNSW index is fast on CPU.
- **Performance impact:**
    - Top-K similarity search (HNSW, CPU):
        - ~5–20 ms @ 50k vectors
        - ~15–60 ms @ 200k vectors
        - ~25–90 ms @ 1M vectors
    - Benchmark: 
        - DB-only vector search: p95 ≈ 2.5 ms (TOPK=10, warm cache) .
        - end-to-end (embedding + vector search): p95 ≈ 9.2 ms.
        - Testbed: AMD Ryzen 7 7700X (8C, 4.50 GHz), 32 GB RAM.
        - Method: 200 iterations (warm-up included), cosine similarity, 384-d embeddings (all-MiniLM-L6-v2), TOPK=10.

- **Storage requirements:**
    - Raw vector payload (float32): 384 dims × 4 bytes = 1,536 bytes ≈ 1.5 KB / vector
    - With index + metadata overhead: typically ~2–3× raw.
    - Estimation: ~3 KB / vector.
        - 10k vectors → ~30 MB
        - 100k vectors → ~300 MB
        - 1M vectors → ~3 GB
- **Cost implications:**
    - $0 per-call for retrieval (self-hosted).
    - $0 for embeddings if Sentence-Transformers runs locally.
    - If using a managed embedder later (e.g., OpenAI) → recurring API cost; not selected for the baseline.
- **Maintenance complexity:**
    - One DB to manage; index creation is declarative.
    - Reindex required if dimension or similarity function changes.
    - Standard backups cover both graph and vector search index information.

- **Alternatives considered:**
    - **TigerGraph:** Great for very large, distributed workloads and heavy graph analytics at scale. Useful if hundreds of millions / billions of nodes/edges are expected.
    - **Amazon Neptune:** Fully managed on AWS. Good if the infrastructure is already on AWS and reduced ops are needed. Trade-offs: cost, vendor lock-in.
    - **ArangoDB:** Multi-model (documents + graph + vectors). Nice if docs + graph together in one engine and flexible modeling is desired.

**Consequences:**
- Performance impact: [9.2]ms query latency
- Storage requirements: [3] GB for [1M] vectors
- Cost implications: $[0]/month for embedding. No API calls cost.
- Maintenance complexity: [Low]

---

### 2. Model Selection & Fine-tuning

**Decision:** Use Sentence-Transformers `all-MiniLM-L6-v2` (384-d) as the embedding model; no fine-tuning for the initial version.
> **Note:** It was decided to keep a single, lazily-initialized embedding model per process using functools.lru_cache(maxsize=1), which avoids repeated instantiation, reduces latency, and keeps memory predictable.

**Context:** Balance between model capability, latency, and cost for embedding generation. It is required strong English retrieval quality with very low latency and a small operational footprint for a local, Docker-friendly setup that indexes and queries via Neo4j’s vector index.

**Embedding Model Selection:**
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, cosine similarity)
- **Rationale:** 
    - Good retrieval quality for short English queries and snippets in the domain.
    - Fast local inference (CPU is enough), simple to run in-process—no separate embedding - service.
    - Compact vectors (384d) keep Neo4j index size and latency low, and simplify scaling.
    - Same model for seeding and online queries → avoids distribution drift.
- **Alternatives Rejected:**
    - `all-MiniLM-L12-v2` (384d): slightly better quality but higher latency/compute; marginal win for the workload.
    - `bge-small-en-v1.5` (384d) / `e5-small-v2` (384d): competitive options; easy drop-ins later if offline eval shows a measurable uplift.
    - OpenAI `text-embedding-3-small/large `(1536d/3072d): strong quality and tooling, but larger vectors (4–8×), external dependency, network latency, and recurring cost.

**Fine-tuning Strategy:**
- [x] Use pre-trained models as-is
- [ ] Later: lightweight domain adaptation (contrastive pairs from queries/solutions)
- [ ] Later: task-specific reranking head if needed

**Performance Benchmarks:**
- Embedding generation: [~6.6]ms per query
- Similarity search: [~2.5]ms for [10] top-k
- Memory footprint: [~70]MB per model instance

**Consequences:**
- Trade-offs: speed and simplicity were priotized over retrieval quality.
- Model update and versioning strategy needed: pin the model name and document any swap (e.g., BGE/E5) behind a flag/env; re-indexing may be required if dimensions change.
- Resource requirements for inference: CPU-only is sufficient for the current scale; GPU can lower embedding latency further if needed.

---

### 3. LLM Integration Decisions

**Decision:** Local-first design with an optional managed LLM only on the slow path. The fast path is LLM-free (graph retrieval → best solution), with a small normalized-query → answer cache.

**Context:** Reliable, low-latency answers and predictable cost are required, while retaining the ability to invoke an LLM for open-ended or multi-hop queries.

**LLM Provider Options:**
- [x] Option A: `GPT-4.1 nano` / OpenAI 
- [ ] Option B: `Gemini 2.5 Flash-Lite` / Google
- [ ] Option C: `Llama 3 8B` / Local Ollama
- [ ] Option D: `GPT-4.1 nano` / Azure OpenAI 

**Alternatives considered:**
- Option A: GPT-4.1 nano / OpenAI — chosen:
    - OpenAI positions 4.1-nano as the fastest and cheapest in the 4.1 family, designed for low-latency workloads with a 1M-token context—a strong fit for our compact Graph-RAG prompts. Pricing (per 1M tokens): $0.10 input / $0.40 output (and $0.025 cached input), delivering very low per-call cost at our prompt sizes.
    - Rationale: 
        1. Ready-to-use account and low integration friction. 
        2. Mature SDK and documentation. 
        3. Adequate quality for  compact Graph-RAG prompts. 
        4. Predictable token pricing.
        5. Easy to replace via a single _call_llm interface.

- Option B: Gemini 2.5 Flash-Lite / Google
    - Google’s “Lite” Flash model targets speed + cost efficiency (per 1M tokens: $0.10 input / $0.40 output, context caching $0.025), likewise optimized for low-latency inference; good choice if we standardize on Google AI Studio / Vertex AI. 

- Option C: Llama 3 8B / Local (Ollama)
    - $0 per call and full data residency; suitable when privacy or offline constraints dominate. On typical CPUs/consumer GPUs, throughput is tens of tokens/s; performance scales with hardware and serving stack.
    - Trade-offs: Ownership of the infrastructure; latency/quality depend on serving device; less “plug-and-play” than managed endpoints.

- Option D: GPT-4.1 nano / Azure OpenAI
    - Choosen model with enterprise controls (RBAC, VNET, regional choices). Pricing is comparable to OpenAI’s public pricing, published per region/zone (e.g., input around $0.11/M, output around $0.44/M). Good fit if the organization standardizes on Azure. 

**Integration Pattern:**
- **Chosen Approach:** Two-tier routing
    - FAST: cache hit, or top-1 similarity > threshold, then fetch best Solution for that Issue and format a deterministic answer (no LLM).
    - SLOW: build graph context (Issues/Solutions/Products, up to 2 hops) and call _call_llm(prompt).

- **Prompt Engineering Strategy:** [Hybrid] → template + structured graph context (primary issues, solutions with effectiveness/success rate, affected products) + concise instructions.
- **Context Window Management:** [Truncation] → compact, structured Graph-RAG prompt with truncation. Only ranked, relevant items are passed to the LLM and each section is kept short to control latency/cost.

    - Sections: Primary Issues, Candidate Solutions, Affected Products, Related Issues, plus brief Answering Guidelines.
    - Ranking/selection: Retrieved information filtered and sorted by Neo4j graph.
    - Truncation: Per-field normalization + char-based cut (_truncate(..., n=220)) for issue descriptions.
    - Step cap: ≤ 6 steps per solution (steps[:6]).

**Fallback Strategy:**
- Fast path already de-risks many queries (cache + deterministic answer). 
- The slow path calls OpenAI directly; explicit retries/timeouts/circuit-breaker are not yet implemented.
- Future work: add request timeouts, limited retries with backoff, and guaranteed graph-only deterministic fallback if the LLM is unavailable or exceeds quotas.

**Consequences:**
- **Response quality vs. latency trade-off:** fast path stays LLM-free and sub-200 ms; slow path adds LLM latency but benefits from structured, small prompts (kept concise via truncation and top-k selection).
- Cost per query: input 0.7K × $0.10/M ≈ $0.00007 + output 0.3K × $0.40/M ≈ $0.00012 → ≈ $0.00019 per call.
- **Vendor lock-in:** mitigated by a single _call_llm integration point.
- **Compliance and data privacy implications:** default local-first flow; when using a managed LLM, we send minimal, low-sensitive graph context. Azure/GCP offer enterprise controls if needed. In case of higher privacy standards needed running Llama 3 8B on-premise with Ollama could be a good fit.

---

### 4. System Design & Architecture

**Decision:** Monolithic, local-first service in Python with two-tier routing (fast/slow paths). Embeddings are generated in-process (LRU-cached model), retrieval runs on Neo4j vector indexes, and the slow path calls a managed LLM. Keep the system synchronous and minimal for the challenge.

**Context:** Simple and reliable design system architecture for fast/slow path routing and scalability required.

**Architecture Pattern:**
- [x] Option A: Monolith: single Python service (API + routing + embedder) talking to Neo4j; OpenAI API consumption for slow path.
- [ ] Option B: Microservices: separate “Retriever” service, “LLM proxy,” and API gateway. Heavier ops; unnecessary for current scope.
- [ ] Option C: Event-driven workers: queue offloads slow path to background workers. Useful for very long tasks; out of scope for v1.
- [ ] Option D: Serverless functions: on-demand for spikes. Could impact our low latency goal.

**Data Flow:**


<p align="center">
  <img src="docs/diagrams/fast-slow-diagram.svg" alt="Fast/Slow routing diagram" width="720" style="max-width:100%;height:auto;">
</p>
Slow Path:
```
Query
  → normalize + cache lookup (hit? → return) → Response (FAST)
  → embed_text(query)
  → Neo4j vector search (idx_issue_embedding, TOPK=10)
  → if top-1 similarity > threshold:
        _best_solution_for_issue(issue_id)
        _format_fast_answer(best)
        cache[query_norm] = answer
        → Response (FAST)
```
Fast Path:
Initiated in case fast path does not find a cache answer or similar previous solution for the query issue.
```
Query
  → embed_text(query)
  → Neo4j vector search (threshold ~0.5)
  → get_graph_context(issue_ids, hops≤2)
  → format_context_for_llm(context, query)  [compact sections + truncation, ≤6 steps/solution]
  → _call_llm(prompt)  (if configured; else deterministic graph answer)
  → Response (SLOW)
```

**Scalability Decisions:**
- One cached embedding model per process → more replicas = more RAM.
- Single Neo4j for graph + vectors.
- Per-process in-memory cache.

**Consequences:**
- Pros: easy to scale, low ops overhead, fast “fast-path” responses.
- Cons: higher RAM as replicas grow; slow path ties up a worker; Neo4j can be a bottleneck under high read load.
- Notes: changing embedding dimension/similarity requires reindexing; swaps within 384d avoid schema changes; LLM cost stays low (short prompts, slow path only).

---

## Implementation Assumptions

1. Language: English-only queries/content (no multilingual handling).
2. Runtime: single process is fine locally; multiple replicas are identical (no shared state beyond Neo4j).
3. Data: seed data matches current schema; IDs (iX, sX, pX) are unique.
4. Failure handling: minimal (happy-path)

---
