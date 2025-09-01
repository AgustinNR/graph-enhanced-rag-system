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
    text = text.strip()
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vec.tolist()

class GraphContextRetriever:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    
    def find_similar_issues(self, query_embedding: List[float], threshold: float = 0.7) -> List[Dict]:
        """
        Find issues similar to the query using Neo4j's vector index (cosine).
        Returns a list of issues with similarity > threshold, sorted desc. 
        Each issue is a dict with: id, title, description, severity, similarity
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
        - solutions: related solutions (1 hop)
        - affected_products: Products affected (1 hop) 
        - related_issues: Other issues solved by same solutions (2 hops)
        
        Return structured context for LLM
        """
        
        context: Dict = {
            "primary_issues": [],
            "solutions": [],
            "affected_products": [],
            "related_issues": []
        }
        
        # Early exit if no issues provided
        if not issue_ids:
            return context
        
        with self.driver.session() as session:
            
            # 1) Primary issues
            rows = session.run(
                """
                MATCH (i:Issue)
                WHERE i.id IN $ids
                RETURN i.id AS id,
                        i.title AS title,
                        i.description AS description,
                        coalesce(i.severity,'unknown') AS severity
                """,
                {"ids": issue_ids},
            )
            primary = [{
                "id": r["id"],
                "title": r["title"],
                "description": r["description"],
                "severity": r["severity"],
            } for r in rows]
            context["primary_issues"] = primary

            if not primary:
                return context  # no valid issues found
            
            # 2) Solutions (1 hop) - add avg effectiveness and supporting issues count
            rows = session.run(
                """
                MATCH (i:Issue)-[r:SOLVED_BY]->(s:Solution)
                WHERE i.id IN $ids
                RETURN s.id   AS id,
                        s.title AS title,
                        s.success_rate AS success_rate,
                        round(avg(coalesce(r.effectiveness, 0.7)), 3) AS avg_effectiveness,
                        count(DISTINCT i) AS supporting_issues
                ORDER BY avg_effectiveness DESC, success_rate DESC, title ASC
                """,
                {"ids": issue_ids},
            )
            context["solutions"] = [{
                "id": r["id"],
                "title": r["title"],
                "success_rate": float(r["success_rate"]) if r["success_rate"] is not None else None,
                "avg_effectiveness": float(r["avg_effectiveness"]),
                "supporting_issues": int(r["supporting_issues"]),
            } for r in rows]
            
            # 3) Affected products (1 hop)
            rows = session.run(
                """
                MATCH (p:Product)-[:HAS_ISSUE]->(i:Issue)
                WHERE i.id IN $ids
                RETURN DISTINCT p.id AS id, p.name AS name
                ORDER BY name ASC
                """,
                {"ids": issue_ids},
            )
            context["affected_products"] = [{
                "id": r["id"],
                "name": r["name"],
            } for r in rows]

            # 4) Related issues
            if max_hops >= 2:
                rows = session.run(
                    """
                    MATCH (i:Issue)-[:SOLVED_BY]->(s:Solution)<-[:SOLVED_BY]-(ri:Issue)
                    WHERE i.id IN $ids AND NOT ri.id IN $ids
                    WITH ri, count(DISTINCT s) AS shared_solutions
                    RETURN ri.id AS id,
                            ri.title AS title,
                            coalesce(ri.severity,'unknown') AS severity,
                            shared_solutions
                    ORDER BY shared_solutions DESC, title ASC
                    """,
                    {"ids": issue_ids},
                )
                context["related_issues"] = [{
                    "id": r["id"],
                    "title": r["title"],
                    "severity": r["severity"],
                    "shared_solutions": int(r["shared_solutions"]),
                } for r in rows]

        return context
    
    def format_context_for_llm(self, context: Dict, query: str) -> str:
        """
        Format the graph context into a prompt for the LLM
        Expects context with keys:
        - primary_issues: [{id,title,description,severity, similarity?}]
        - solutions: [{id,title,success_rate?,avg_effectiveness?,supporting_issues?,steps?}]
        - affected_products: [{id,name}]
        - related_issues: [{id,title,severity,shared_solutions?}]
        """

        def _truncate(s: str, n: int = 220) -> str:
            """
            Normalize whitespace and truncate a string to n chars, adding "..." if needed.
            """
            
            if not s:
                return ""
            s = " ".join(s.split())
            return s if len(s) <= n else s[: n - 3] + "..."

        def _fmt_issue(i: Dict) -> str:
            """
            Expected output format example:
            - [i1] Issue title (severity: high) — similarity: 0.873
            description: Truncated description...
            """
            
            line = f"- [{i.get('id','?')}] {i.get('title','')}"
            sev = i.get("severity")
            if sev:
                line += f" (severity: {sev})"
            sim = i.get("similarity")
            if isinstance(sim, (int, float)):
                line += f" — similarity: {sim:.3f}"
            desc = _truncate(i.get("description", ""))
            if desc:
                line += f"\n  description: {desc}"
            return line

        def _fmt_solution(s: Dict) -> str:
            
            """
            Expected output format example:
            - [s1] Solution title — avg_effectiveness: 0.850 — success_rate: 0.92 — supports: 3 issues
            steps: Step 1; Step 2; Step 3...
            """
            
            parts = [f"- [{s.get('id','?')}] {s.get('title','')}"]
            if s.get("avg_effectiveness") is not None:
                parts.append(f"avg_effectiveness: {float(s['avg_effectiveness']):.2f}")
            if s.get("success_rate") is not None:
                parts.append(f"success_rate: {float(s['success_rate']):.2f}")
            if s.get("supporting_issues"):
                parts.append(f"supports: {int(s['supporting_issues'])} issues")
            line = " — ".join(parts)
            steps = s.get("steps")
            if steps:
                line += "\n  steps: " + "; ".join(steps[:6])
            return line

        def _fmt_product(p: Dict) -> str:
            """
            Expected output format example:
            - [p1] Product name
            """
            
            return f"- [{p.get('id','?')}] {p.get('name','')}"

        def _fmt_related(i: Dict) -> str:
            
            """
            Expected output format example:
            - [i2] Issue title (severity: medium) — shared_solutions: 2 
            """
            
            line = f"- [{i.get('id','?')}] {i.get('title','')}"
            sev = i.get("severity")
            if sev:
                line += f" (severity: {sev})"
            if i.get("shared_solutions"):
                line += f" — shared_solutions: {int(i['shared_solutions'])}"
            return line

        prim = context.get("primary_issues", []) or []
        sols = context.get("solutions", []) or []
        prods = context.get("affected_products", []) or []
        rels = context.get("related_issues", []) or []

        sections = []

        # Header + user query
        sections.append(
            f"User Query: {query}\n"
            "Main Intructions:\n"
            "You are a helpful technical assistant. Use ONLY the context below."
            "If the context is insufficient or ambiguous, apologize and say you don't know.\n"
        )

        # Primary issues
        if prim:
            sections.append("Primary Issues:\n" + "\n".join(_fmt_issue(i) for i in prim))
        else:
            sections.append("Primary Issues:\n- (none)")

        # Solutions
        if sols:
            sections.append("Candidate Solutions (ranked):\n" + "\n".join(_fmt_solution(s) for s in sols))
        else:
            sections.append("Candidate Solutions:\n- (none)")

        # Affected products
        if prods:
            sections.append("Affected Products:\n" + "\n".join(_fmt_product(p) for p in prods))

        # Related issues (2 hops)
        if rels:
            sections.append("Related Issues (via shared solutions):\n" + "\n".join(_fmt_related(i) for i in rels))

        # Output guidelines for the LLM
        sections.append(
            "Answering Guidelines:\n"
            "- Prefer solutions with higher avg_effectiveness and success_rate; consider issue severity.\n"
            "- Reference solutions by their IDs (e.g., [s1]) when recommending steps.\n"
            "- Provide a concise, step-by-step plan.\n"
            "- If multiple solutions apply, compare briefly and pick one to try first."
        )

        prompt = "\n\n".join(sections)
        return prompt