import asyncio
from typing import Tuple, Dict, List, Optional
from enum import Enum
from openai import OpenAI
import os
from dotenv import load_dotenv
from src.context_retriever import embed_text, GraphContextRetriever

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ResponsePath(Enum):
    FAST = "fast"  # Cached or high-similarity match
    SLOW = "slow"  # Needs graph traversal + LLM

class SmartResponseSystem:
    def __init__(self, context_retriever: GraphContextRetriever):
        self.retriever = context_retriever
        self.similarity_cache: Dict[str, str] = {
            "how do i reset my device?": "To reset your device, press and hold the power button for 10 seconds until it restarts.",
            "what's the warranty period?": "The standard warranty period is 12 months from the date of purchase.",
        }  # cache_key -> answer - Simple cache for demo
        #self.fast_threshold = 0.85  # High similarity = fast path
        self.fast_threshold = 0.75  # High similarity = fast path
        
    async def process_query(self, query: str) -> Tuple[str, ResponsePath]:
        """
        Main entry point - routes to fast or slow path
          - Cache hit -> FAST
          - Top-1 similarity >= fast_threshold -> FAST (+ caching)
          - Else -> SLOW (complete graph traversal + LLM)    
        """
        
        # 1) Cache check first
        cache_key = self._get_cache_key(query)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key], ResponsePath.FAST
        
        # 2) Check similarity to known issues
        query_embedding = embed_text(query)
        similar_issues = self.retriever.find_similar_issues(query_embedding, threshold=self.fast_threshold)
        # check if the list is not empty
        if similar_issues:
            top_issue_id = similar_issues[0]["id"]
            return await self._fast_path(cache_key, top_issue_id)

        # 3) Else, slow path
        return await self._slow_path(query_embedding, query)
    
    async def _fast_path(self, cache_key: str, issue_id: str) -> Tuple[str, ResponsePath]:
        """
        Fast path: Use cached context or high-similarity match
        Should respond in < 200ms
        """
        # Look for the best solution for the issue
        best = self._best_solution_for_issue(issue_id)
        answer = self._format_fast_answer(best)
        self.similarity_cache[cache_key] = answer
        return answer, ResponsePath.FAST        
    
    async def _slow_path(self, query_embedding: List[float], query: str) -> Tuple[str, ResponsePath]:
        """
        Slow path: Full graph traversal + LLM generation
        Can take 2-5 seconds
        """
    
        similar_issues = self.retriever.find_similar_issues(query_embedding, threshold=0.5)
        context = self.retriever.get_graph_context(similar_issues, max_hops=2)
        prompt = self.retriever.format_context_for_llm(context, query)
        answer = self._call_llm(prompt)
        return answer, ResponsePath.SLOW

    def _call_llm(self, prompt: str, model: str = "gpt-4.1-nano") -> str:
        """
        Call your LLM of choice
        Note: Can use OpenAI, Hugging Face, Ollama, or mock for testing
        """
        # IMPLEMENT: LLM call
        # For testing, you can mock this:
        # return f"Mock LLM response for: {prompt[:100]}..."
        client = OpenAI()

        response = client.responses.create(
            model=model,
            input=prompt
        )
        
        return response.output_text
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        # Simple implementation - in production, use better hashing
        return query.lower().strip()
    
    def _best_solution_for_issue(self, issue_id: str) -> Optional[Dict]:
        """
        Returns a dict with information of the best Solution for the given issue_id.
        """
        with self.retriever.driver.session() as s:
            row = s.run(
                """
                MATCH (i:Issue {id:$id})-[r:SOLVED_BY]->(s:Solution)
                OPTIONAL MATCH (p:Product)-[:HAS_ISSUE]->(i)
                WITH i, s,
                        avg(coalesce(r.effectiveness,0.7)) AS avg_eff,
                        s.success_rate AS sr,
                        collect(DISTINCT p.name) AS products
                RETURN i.id AS issue_id,
                        i.title AS issue_title,
                        coalesce(i.severity,'unknown') AS severity,
                        s.id AS sol_id, s.title AS sol_title, s.steps AS steps,
                        round(avg_eff,3) AS avg_effectiveness, sr AS success_rate,
                        products
                ORDER BY avg_eff DESC, sr DESC, sol_title ASC
                LIMIT 1
                """,
                {"id": issue_id},
            ).single()

        if not row:
            return None
        return {
            "issue_id": row["issue_id"],
            "issue_title": row["issue_title"],
            "severity": row["severity"],
            "sol_id": row["sol_id"],
            "sol_title": row["sol_title"],
            "steps": row["steps"] or [],
            "avg_effectiveness": row["avg_effectiveness"],
            "success_rate": row["success_rate"],
            "products": row["products"] or [],
        }

    def _format_fast_answer(self, best: Dict) -> str:
        steps = best["steps"]
        steps_txt = "\n".join(f"  {i+1}. {st}" for i, st in enumerate(steps)) if steps else "No steps available."
        prods = best["products"]
        prods_line = f"\nAffected product(s): {', '.join(prods)}." if prods else ""
        metrics = []
        if best["avg_effectiveness"] is not None:
            metrics.append(f"avg effectiveness: {best['avg_effectiveness']:.2f}")
        if best["success_rate"] is not None:
            metrics.append(f"success rate: {best['success_rate']:.2f}")
        metrics_line = f" ({'; '.join(metrics)})" if metrics else ""

        return (
            f"It looks like **{best['issue_title']}** (severity: {best['severity']}).{prods_line}\n\n"
            f"Try **[{best['sol_id']}] {best['sol_title']}**{metrics_line}:\n{steps_txt}\n\n"
            f"If this doesn’t solve it, try with a new question."
            )