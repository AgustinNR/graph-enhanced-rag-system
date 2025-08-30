# 🎯 AI Engineer Graph-Enhanced RAG System

A context retrieval system that combines Neo4j knowledge graphs with embeddings and LLM integration for technical support questions.

## 📁 Project Structure

```
aichallenge_clean/
├── 📁 scripts/                # Utility scripts
│   └── 📄 seed_data.py        # Database seeding
├── 📁 src/                    # Main implementation package
│   ├── 📄 __init__.py
│   ├── 📄 context_retriever.py # Graph context retrieval
│   └── 📄 response_system.py   # LLM integration & routing
├── 📁 tests/                   # Test suite
│   ├── 📄 __init__.py
├── 📄 main.py                 # Main python file for run
├── 📄 .env                    # Environment variables template
├── 📄 ARCHITECTURE.md         # Architecture Decision Record
├── 📄 CHALLENGE.md            # Challenge instructions
├── 📄 docker-compose.yml      # Neo4j database setup
├── 📄 pyproject.toml          # Project configuration with uv
└── ...
```

## 🔧 Prerequisites

### Required Tools
- **Docker & Docker Compose** - For Neo4j database
- **uv** - Fast Python package manager ([Install guide](uv)
- **Python 3.13** - Programming language ([Install guide thought uv](https://docs.astral.sh/uv/getting-started/features/))

## 🚀 Setup Instructions

⚠️ **Important**: You must implement the `embed_text()` function before running the seed script.

```bash
# 1. Setup environment
uv sync

# 2. Start Neo4j
docker-compose up -d

# Neo4j will be available at:
# Browser: http://localhost:7474
# Bolt: bolt://localhost:7687
# Credentials: neo4j/password

# Check status: 
docker-compose ps

# 3. Test it
uv run python main.py

# 4. Implement embeddings in src/context_retriever.py first
# The embed_text() function is required for seeding

# 5. Seed database 
# uv run python scripts/seed_data.py
```

## 🎯 Challenge Instructions

For detailed challenge instructions and requirements, see [CHALLENGE.md](CHALLENGE.md).
