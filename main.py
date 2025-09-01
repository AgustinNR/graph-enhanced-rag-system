# test_system.py - PROVIDED
import asyncio
import os
import logging
from src.context_retriever import GraphContextRetriever
from src.response_system import SmartResponseSystem
from dotenv import load_dotenv
from src.context_retriever import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

async def main():
    load_dotenv()
    # Initialize your system
    retriever = GraphContextRetriever(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    system = SmartResponseSystem(retriever)
    
    # --- warmup (keep the first request fast) ---
    _ = embed_text("warmup")  # loads model/tokenizer

    # Neo4j handshake (cheap)
    with retriever.driver.session() as s: s.run("RETURN 1").consume()
    # --- end warmup ---

    test_queries = [
        "Group pairing of multiple RGB bulbs fails intermittently", # Should find similar issue with high threshold - route=FAST source=vector
        "App cannot find the device during setup",  # Should find similar issue - route=FAST source=vector
        "My SmartHub won't connect to WiFi",  # Should find similar issue (highly affected by threshold) - route=FAST source=vector / route=SLOW
        "The hub disconnects after exactly 2 hours and only when three lights are connected",  # Complex - route=SLOW 
        "How do I reset my device?",  # Common query - route=FAST source=cache
        "What's the warranty period?",  # Simple FAQ - route=FAST source=cache
        "Group pairing of multiple RGB bulbs fails intermittently", # Repeat to test caching - route=FAST source=cache
    ]
    
    for query in test_queries:
        response, path = await system.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Path: {path.value}")
        #print(f"Response: {response[:200]}...")
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())