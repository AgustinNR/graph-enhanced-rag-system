from src.context_retriever import embed_text

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
    }
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
    }
]

solutions = [
    {
        "id": "s1",
        "title": "Network Reset Procedure",
        "steps": ["Power cycle device", "Hold reset button 10 seconds", "Reconfigure WiFi"],
        "embedding": embed_text("Reset network settings power cycle WiFi reconfigure"),
        "success_rate": 0.85
    }
]

# Cypher to load this data
load_query = """
// Create products
UNWIND $products as p
CREATE (prod:Product {
    id: p.id,
    name: p.name,
    description: p.description,
    embedding: p.embedding
})

// Create issues  
UNWIND $issues as i
CREATE (issue:Issue {
    id: i.id,
    title: i.title,
    description: i.description,
    embedding: i.embedding,
    severity: i.severity
})

// Create solutions
UNWIND $solutions as s
CREATE (sol:Solution {
    id: s.id,
    title: s.title,
    steps: s.steps,
    embedding: s.embedding,
    success_rate: s.success_rate
})

// Create relationships
MATCH (p:Product {id: 'p1'}), (i:Issue {id: 'i1'})
CREATE (p)-[:HAS_ISSUE]->(i)

MATCH (i:Issue {id: 'i1'}), (s:Solution {id: 's1'})
CREATE (i)-[:SOLVED_BY {effectiveness: 0.9}]->(s)
"""