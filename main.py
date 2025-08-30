# test_system.py - PROVIDED
import asyncio
import os

from src.context_retriever import GraphContextRetriever
from src.response_system import SmartResponseSystem
from dotenv import load_dotenv

async def main():
    load_dotenv()
    # Initialize your system
    retriever = GraphContextRetriever(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    system = SmartResponseSystem(retriever)
    
    test_queries = [
        "My SmartHub won't connect to WiFi",  # Should find similar issue
        "How do I reset my device?",  # Common query - fast path
        "The hub disconnects after exactly 2 hours and only when three lights are connected",  # Complex - slow path
        "What's the warranty period?",  # Simple FAQ - fast path
    ]
    
    for query in test_queries:
        response, path = await system.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Path: {path.value}")
        print(f"Response: {response[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())