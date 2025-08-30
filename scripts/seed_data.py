import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from src.context_retriever import embed_text

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Sample data with embeddings feel free to add more data
products = [
    {
        "id": "p1",
        "name": "SmartHub Pro",
        "description": "Central hub for smart home devices with WiFi and Zigbee",
        "embedding": embed_text("Central hub for smart home devices with WiFi and Zigbee connectivity")
    },
    {
        "id": "p2", 
        "name": "SmartLight RGB",
        "description": "Color changing smart bulb with app control",
        "embedding": embed_text("Color changing smart bulb with app control")
    },
    {
        "id": "p3",
        "name": "SmartCamera",
        "description": "WiFi security camera with motion detection and night vision",
        "embedding": embed_text("WiFi security camera for home monitoring with motion detection and night vision")
    },
    {
        "id": "p4",
        "name": "SmartThermostat",
        "description": "Smart thermostat with schedule learning and energy saving",
        "embedding": embed_text("Smart thermostat that learns schedules and optimizes energy consumption")
    },
]

issues = [
    {
        "id": "i1",
        "title": "WiFi connection drops",
        "description": "Device loses WiFi connection intermittently",
        "embedding": embed_text("Device loses WiFi connection intermittently"),
        "severity": "high"
    },
    {
        "id": "i2",
        "title": "Cannot find device in app",
        "description": "Mobile app cannot detect or find the device during setup",
        "embedding": embed_text("Mobile app cannot detect or find the device during setup"),
        "severity": "high"
    },
    {
        "id": "i3",
        "title": "Bulbs not pairing in groups",
        "description": "Group pairing of multiple RGB bulbs fails intermittently",
        "embedding": embed_text("Multiple RGB bulbs fail to pair as a group during setup"),
        "severity": "medium"
    },
    {
        "id": "i4",
        "title": "Thermostat schedule not applying",
        "description": "Smart thermostat does not apply learned schedule after firmware update",
        "embedding": embed_text("Smart thermostat ignores learned schedule after a firmware update"),
        "severity": "low"
    },
]

solutions = [
    {
        "id": "s1",
        "title": "Network Reset Procedure",
        "steps": ["Power cycle device", "Hold reset button 10 seconds", "Reconfigure WiFi"],
        "embedding": embed_text("Reset network settings power cycle WiFi reconfigure"),
        "success_rate": 0.85
    },
    {
        "id": "s2",
        "title": "Firmware Update",
        "steps": ["Open app", "Check for firmware update", "Apply update", "Reboot device"],
        "embedding": embed_text("Update device firmware via app then reboot"),
        "success_rate": 0.65
    },
    {
        "id": "s3",
        "title": "Router Configuration Check",
        "steps": ["Ensure 2.4GHz enabled", "Disable band steering", "Reserve IP via DHCP"],
        "embedding": embed_text("Verify router settings 2.4GHz disable band steering assign static DHCP reservation"),
        "success_rate": 0.72
    },
]

rels_product_issue = [
    {"p": "p1", "i": "i1"},
    {"p": "p2", "i": "i1"},
    {"p": "p2", "i": "i3"},
    {"p": "p3", "i": "i1"},
    {"p": "p4", "i": "i4"},
]

rels_issue_solution = [
    {"i": "i1", "s": "s1", "effectiveness": 0.90},
    {"i": "i1", "s": "s3", "effectiveness": 0.75},
    {"i": "i2", "s": "s1", "effectiveness": 0.80},
    {"i": "i2", "s": "s2", "effectiveness": 0.70},
    {"i": "i3", "s": "s1", "effectiveness": 0.72},
    {"i": "i4", "s": "s2", "effectiveness": 0.66},
]

create_indexes = [
    """
    CREATE VECTOR INDEX idx_product_embedding IF NOT EXISTS
    FOR (p:Product) ON (p.embedding)
    OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } };
    """,
    """
    CREATE VECTOR INDEX idx_issue_embedding IF NOT EXISTS
    FOR (i:Issue) ON (i.embedding)
    OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } };
    """,
    """
    CREATE VECTOR INDEX idx_solution_embedding IF NOT EXISTS
    FOR (s:Solution) ON (s.embedding)
    OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } };
    """,
]

# Cypher to load this data
load_query = """
// Products
UNWIND $products AS p
MERGE (prod:Product {id: p.id})
  SET prod.name = p.name,
      prod.description = p.description,
      prod.embedding = p.embedding
WITH 1 AS _

// Issues
UNWIND $issues AS i
MERGE (issue:Issue {id: i.id})
  SET issue.title = i.title,
      issue.description = i.description,
      issue.embedding = i.embedding,
      issue.severity = i.severity
WITH 1 AS _

// Solutions
UNWIND $solutions AS s
MERGE (sol:Solution {id: s.id})
  SET sol.title = s.title,
      sol.steps = s.steps,
      sol.embedding = s.embedding,
      sol.success_rate = s.success_rate
WITH 1 AS _

// Product -> Issue
UNWIND $rels_pi AS rpi
MATCH (p:Product {id: rpi.p}), (i:Issue {id: rpi.i})
MERGE (p)-[:HAS_ISSUE]->(i)
WITH 1 AS _

// Issue -> Solution (with effectiveness attribute)
UNWIND $rels_is AS ris
MATCH (i:Issue {id: ris.i}), (s:Solution {id: ris.s})
MERGE (i)-[rel:SOLVED_BY]->(s)
  SET rel.effectiveness = coalesce(ris.effectiveness, 0.7)
"""


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Create vector indexes
        print("Creating indexes...")
        for stmt in create_indexes:
            try:
                session.run(stmt)
            except Exception as e:
                print("Index creation notice:", e)

        # Load data
        print("Loading seed data...")
        session.run(
            load_query,
            {
                "products": products,
                "issues": issues,
                "solutions": solutions,
                "rels_pi": rels_product_issue,
                "rels_is": rels_issue_solution,
            },
        )

        # Sample query to verify data load
        res = session.run("""
        MATCH (p:Product)-[:HAS_ISSUE]->(i:Issue)-[:SOLVED_BY]->(s:Solution)
        RETURN p.id AS product, i.id AS issue, s.id AS solution
        LIMIT 10
        """)
        rows = list(res)
        print("Sample relationships loaded:")
        for r in rows:
            print(r["product"], "->", r["issue"], "->", r["solution"])

    driver.close()
    print("✅ Seed completed.")

if __name__ == "__main__":
    main()