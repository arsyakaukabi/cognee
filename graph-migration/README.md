# Cognee Graph Migration (Kuzu -> Neo4j)

This folder contains a complete, repeatable migration pipeline to move Cognee graph data
from Kuzu to Neo4j. It exports Kuzu data to CSV, transforms it into Neo4j-ready CSVs,
imports into Neo4j, and validates the result.

## Prerequisites

- Python 3.10+ (aligned with Cognee)
- Kuzu database path accessible locally
- Neo4j 5.x with APOC plugin enabled
- Neo4j configured to allow file CSV import

Optional:
- Docker (to run Neo4j locally using `docker-compose.yml`)

## Quickstart

1) Install dependencies

```bash
python -m pip install -r graph-migration/requirements.txt
```

2) Configure environment

```bash
cp graph-migration/.env.example graph-migration/.env
# edit graph-migration/.env
```

3) (Optional) Start Neo4j with Docker

```bash
cd graph-migration
docker compose up -d
```

4) Run end-to-end migration

```bash
./graph-migration/scripts/run_all.sh
```

Reports are written to `graph-migration/output/reports`.

## How it works

- `scripts/export_kuzu.py` exports Kuzu `Node` and `EDGE` tables to raw CSVs
- `scripts/transform_to_neo4j.py` normalizes data, splits by type, generates Cypher
- `scripts/import_neo4j.py` creates schema and imports CSVs into Neo4j
- `scripts/validate_neo4j.py` runs post-import checks and outputs a report

## Notes

- The import uses `LOAD CSV` and APOC JSON conversion. Ensure Neo4j allows file imports
  and the CSVs are in Neo4j's import directory. The provided Docker compose mounts
  `graph-migration/output/neo4j_csv` into `/import`.
- The pipeline is idempotent for nodes and relationships (MERGE based on ids and types).
- By default, the import refuses to run if the Neo4j database is not empty. You can
  override with `NEO4J_ALLOW_NONEMPTY=true` or wipe it with `NEO4J_CLEAR_DATABASE=true`.

## Full runbook

See `docs/04-migration-runbook.md` for step-by-step instructions and rollback guidance.
