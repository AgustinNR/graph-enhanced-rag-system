# AI Engineer Code Challenge: Graph-Enhanced RAG System

⏱️ **Time Limit: 2 Hours** | 🎯 **Focus: System Design & Implementation**

> **Important** 
> - Read this document until the end before starting the challenge.
> - Using AI Assistants is discouraged as we will discuss your solution in later interviews.

---

## 1. Overview

### What You're Building
A context retrieval system that combines Neo4j knowledge graphs with embeddings and LLM integration for technical support questions.

### Challenge Objective
Build a context retrieval system that combines Neo4j knowledge graphs with embeddings and LLM integration to answer technical support questions. The system should demonstrate smart context retrieval and fast/slow path routing based on semantic similarity.

### What We're Evaluating

1. **Embeddings & Semantic Search** - Can you use embeddings to find relevant information?
2. **Graph Context Retrieval** - Can you traverse a knowledge graph to build rich context?
3. **LLM Integration** - Can you effectively prompt an LLM with retrieved context?
4. **Async Architecture** - Can you design fast response systems?

### The Scenario

You're building an AI support assistant that:
- Fetches context from a knowledge graph
- Feeds context to an LLM for response generation
- Focus on fast responses, even when the graph is large and queries are complex
- The goal is to answer questions in the millisecond to one second range
- The knowledge graph can potentially contain billions of nodes and edges

---

## 2. Implementation Tasks

### Part 1: Embeddings & Graph Context Retrieval

You will implement a multihop retrieval system that fetches the relevant context from a knowledge graph.

#### 1.1 Provided Neo4j Schema

```cypher
# The graph contains:
# - Products with description embeddings
# - Issues with title and description embeddings  
# - Solutions with embedded step descriptions
# - Relationships showing which solutions work for which issues

# Nodes have this structure:
Product {
    id: str,
    name: str,
    description: str,
    embedding: List[float]  # 384-dim embedding from sentence-transformers
}

Issue {
    id: str,
    title: str,
    description: str,
    embedding: List[float],
    severity: str
}

Solution {
    id: str,
    title: str,
    steps: List[str],
    embedding: List[float],
    success_rate: float
}

(p:Product)-[:HAS_ISSUE]->(i:Issue)
(i:Issue)-[:SOLVED_BY]->(s:Solution)
```

#### 1.2 Your Task: Implement Context Retrieval

See `src/context_retriever.py` for the implementation template with the following methods:
- `embed_query()` - Convert user query to embedding
- `find_similar_issues()` - Find similar issues using cosine similarity
- `get_graph_context()` - Traverse graph to gather rich context
- `format_context_for_llm()` - Format context for LLM prompting

### Part 2: LLM Integration & Fast/Slow Path

#### 2.1 Implement the Response System

Use the implemented retrieved context from part 1 to solve the user's issue. You might need to cache similar issues so there is no need to perform the retrieval to respond to the user.

See `src/response_system.py` for the implementation template with the following components:
- `ResponsePath` enum for fast/slow routing
- `SmartResponseSystem` class with async query processing
- `process_query()` - Main entry point with routing logic

#### 2.2 System Architecture Design

Use the `architecture.md` file to create an Architecture Decision Record covering:
- Embedding strategy and storage decisions
- Model selection and fine-tuning decisions
- LLM integration decisions
- System design decisions
- Caching decisions
- Performance metrics and tracking decisions

---

## 3. Resources & Testing

### Provided Starter Data

See `scripts/seed_data.py` for the complete dataset including:
- Sample products (SmartHub Pro, SmartLight RGB) with embeddings
- Common issues (WiFi drops, device detection) with embeddings
- Solutions (Network reset procedures) with success rates
- Cypher queries to load data into Neo4j with proper relationships

### (Optional) Testing Your Implementation

We provided a test suite to help you verify your implementation. Still, if your code doesn't work, don't expend time on fixing it. Focus on the main goals. The main goal is to understand your knowledge regarding designing and implementing LLM-based systems. This exercise will serve as a reference for the next interviews using screen sharing.

#### Provided Test Cases

See `main.py` for the complete test suite with:
- System initialization with Neo4j connection
- Four test queries covering different complexity levels:
  - WiFi connectivity issues (similarity matching)
  - Device reset procedures (fast path)
  - Complex multi-condition problems (slow path)
  - Simple FAQ queries (fast path)
- Async test runner with response path tracking

---

## 4. Deliverables & Evaluation

### Deliverables Checklist

#### Required (Must Complete)
- [ ] `src/context_retriever.py` - Embedding search and graph context retrieval
- [ ] `src/response_system.py` - Fast/slow path routing with LLM integration
- [ ] `docs/architecture.md` - ADR (Architecture Decision Record) for system design

#### Bonus (If Time Permits)
- [ ] Implement embedding caching strategy
- [ ] Add similarity threshold tuning
- [ ] Performance metrics (latency tracking)

### Evaluation Criteria

#### AI/ML Understanding (40%)
- Proper use of embeddings for semantic search
- Effective prompt engineering for LLM
- Understanding of when to use different models/approaches

#### Graph Integration (20%)
- Efficient context retrieval from Neo4j
- Smart traversal strategies
- Combining embeddings with graph structure

#### System Design (40%)
- ADR is clearly documented and well-written
- All your assumptions are valid and justified
- All the decisions are included (even those that are not relevant for the final solution in case you changed a decision)
- All decisions are justified according to the requirements and assumptions

---

## 5. Guidelines & Support

### 💡 Tips

1. **LLM Integration**: You can mock the LLM calls if you don't have API access. We care more about the integration pattern than actual responses.

2. **Embeddings**: Use the smallest and fastest model that comply with the requirements. You can use `sentence-transformers` for Python or `transformers` for Hugging Face. You can also use `openai` for Python or `ollama` for Hugging Face. You can also use `numpy` for vector operations.

3. **Neo4j Vectors**: Neo4j 5.x supports vector indexes. You can use `db.index.vector.queryNodes()` or compute similarity in Python.

4. **Time Management**:
   - This challenge is designed to be imperfect, so focus on the main goals and avoid expending too much time on details that are not relevant to the main goal (unnecessary refactors)
   - This challenge was designed to be completed in about two hours, so make sure to spend time on the most important parts of the system. Don't expend more than that
   - If your code doesn't work, don't expend time on fixing it. Focus on the main goals

### ⚠️ What We're NOT Expecting

- Production-ready code
- Complex caching mechanisms
- Fine-tuned models
- Perfect LLM responses
- Extensive error handling
- Even working code (we care more about the design)

### 🚀 Getting Started

1. Make sure you've completed the setup instructions in [README.md](README.md)
2. Start with `src/context_retriever.py`
3. Then implement `src/response_system.py`
4. Document your decisions in `docs/architecture.md`
5. Test your implementation with `tests/test_system.py`

### 🤖 LLM Support Options

You can choose the one that best fits your needs:

- **OpenAI** (cloud-based)
- **Ollama** (local runtime for LLMs, optional)
- **Hugging Face / Transformers** (local or custom models)
- **Any other provider** you wish to integrate

#### 🐳 Running Local LLMs with Docker

If you choose to use a **local LLM provider** (such as Ollama or Hugging Face/Transformers),  
it is **mandatory** to provide a Docker setup that runs the model and exposes it on a port.  

This is required so that we can easily start the container and verify that the project works correctly during testing.

If additional steps are needed, you may include them in a separate file
(e.g., INSTRUCTIONS.md) with detailed build/run instructions for your Docker setup.

### ❓ Questions?

**Focus Areas**: Embeddings usage, context retrieval strategy, and smart routing logic!

Make reasonable assumptions and document them. We value your approach and thinking process over perfect implementation.

---

🎆 **Good luck with the challenge!** Remember to focus on system design and document your decisions in the ADR.
