import asyncio
from typing import Tuple, Dict
from enum import Enum
# import openai  # or use Hugging Face, Ollama, etc.

from .context_retriever import embed_text


class ResponsePath(Enum):
    FAST = "fast"  # Cached or high-similarity match
    SLOW = "slow"  # Needs graph traversal + LLM

class SmartResponseSystem:
    def __init__(self, context_retriever):
        self.retriever = context_retriever
        self.similarity_cache = {}  # Simple cache for demo
        self.fast_threshold = 0.85  # High similarity = fast path
        
    async def process_query(self, query: str) -> Tuple[str, ResponsePath]:
        """
        Main entry point - routes to fast or slow path
        """
        # IMPLEMENT: Decision logic for fast vs slow path
        query_embedding = embed_text(query)
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.similarity_cache:
            return await self._fast_path(query, self.similarity_cache[cache_key])
        
        # Check similarity to known issues
        similar_issues = self.retriever.find_similar_issues(query_embedding)
        
        # YOUR ROUTING LOGIC HERE
        # When to use fast path vs slow path?
        
        return "Hello world", ResponsePath.FAST
    
    async def _fast_path(self, query: str, cached_context: Dict) -> Tuple[str, ResponsePath]:
        """
        Fast path: Use cached context or high-similarity match
        Should respond in < 200ms
        """
        # IMPLEMENT: Quick response using cached/similar context
        # Can use a small LLM or even template responses
        pass
    
    async def _slow_path(self, query: str) -> Tuple[str, ResponsePath]:
        """
        Slow path: Full graph traversal + LLM generation
        Can take 2-5 seconds
        """
        # IMPLEMENT: 
        # 1. Get query embedding
        # 2. Find similar issues
        # 3. Traverse graph for context
        # 4. Call LLM with context
        # 5. Cache the result
        pass
    
    def _call_llm(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Call your LLM of choice
        Note: Can use OpenAI, Hugging Face, Ollama, or mock for testing
        """
        # IMPLEMENT: LLM call
        # For testing, you can mock this:
        # return f"Mock LLM response for: {prompt[:100]}..."
        pass
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        # Simple implementation - in production, use better hashing
        return query.lower().strip()