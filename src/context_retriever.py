from typing import List, Dict
from neo4j import GraphDatabase
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # lazy load to avoid re-instantiating the model on each call
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str) -> List[float]:
    """
    Convert text into a 384-dim embedding using all-MiniLM-L6-v2.
    Returns a Python list[float] so it can be stored directly in Neo4j.
    """
    text = (text or "").strip()
    if not text:
        return [0.0] * 384
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()

class GraphContextRetriever:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    
    def find_similar_issues(self, query_embedding: List[float], threshold: float = 0.7) -> List[Dict]:
        """
        Find issues similar to the query using Neo4j's vector index (cosine).
        Returns issues with similarity > threshold, sorted desc.
        """
        INDEX_NAME = "idx_issue_embedding"  # Ensure this matches your Neo4j index name
        K = 10                              # max number of neighbors to retrieve
        
        cypher = """
        CALL db.index.vector.queryNodes($index, $k, $qemb)
        YIELD node, score
        WITH node, score
        WHERE score > $threshold
        RETURN node.id AS id,
            node.title AS title,
            node.description AS description,
            coalesce(node.severity, 'unknown') AS severity,
            score AS similarity
        ORDER BY similarity DESC
        """
        # EXAMPLE: Enhanced ordering by severity (if needed)
        # You can uncomment and modify the below if you want to prioritize by severity as well
        #   cypher = """
        #   CALL db.index.vector.queryNodes($index, $k, $qemb)
        #   YIELD node, score
        #   WITH node, score
        #   WHERE score > $threshold
        #    RETURN node.id AS id, node.title AS title, node.description AS description,
        #       coalesce(node.severity,'unknown') AS severity,
        #       CASE toLower(node.severity)
        #               WHEN 'high'   THEN 3
        #               WHEN 'medium' THEN 2
        #               WHEN 'low'    THEN 1
        #               ELSE 0
        #       END AS sev_rank,
        #       score AS similarity
        #   ORDER BY similarity DESC, sev_rank DESC
        #   """
        
        with self.driver.session() as session:  
            res = session.run(
                cypher,
                {
                    "index": INDEX_NAME,
                    "qemb": query_embedding,
                    "k": K,
                    "threshold": float(threshold),
                },
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "severity": r["severity"],
                    "similarity": float(r["similarity"]),
                }
            for r in res
        ]
    
    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass
    
    def get_graph_context(self, issue_ids: List[str], max_hops: int = 2) -> Dict:
        """
        Given issue IDs, traverse the graph to get:
        - Related solutions (1 hop)
        - Products affected (1 hop) 
        - Other issues solved by same solutions (2 hops)
        
        Return structured context for LLM
        """
        with self.driver.session() as session:
            # IMPLEMENT: Build context from graph traversal
            context = {
                "primary_issues": [],
                "solutions": [],
                "affected_products": [],
                "related_issues": []
            }
            
            # YOUR IMPLEMENTATION HERE
            # Traverse the graph to build rich context
            
            return context
    
    def format_context_for_llm(self, context: Dict, query: str) -> str:
        """
        Format the graph context into a prompt for the LLM
        """
        # IMPLEMENT: Create a well-structured prompt
        prompt = f"""
        User Query: {query}
        
        Based on our knowledge graph, here's the relevant context:
        
        // YOUR FORMATTING HERE
        
        Please provide a helpful response using this context.
        """
        return prompt