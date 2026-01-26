#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    raise SystemExit("neo4j package is required. Install with: pip install -r graph-migration/requirements.txt") from exc

from lib import load_env, getenv, log


def split_statements(text: str):
    statements = []
    buff = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        buff.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buff).strip().rstrip(";").strip()
            if stmt:
                statements.append(stmt)
            buff = []
    if buff:
        stmt = "\n".join(buff).strip()
        if stmt and not stmt.strip().startswith("//"):
            statements.append(stmt)
    return statements


def run_cypher_file(session, path: Path):
    text = path.read_text(encoding="utf-8")
    statements = split_statements(text)
    for idx, stmt in enumerate(statements, start=1):
        log(f"Executing statement {idx}/{len(statements)} from {path.name}")
        session.run(stmt).consume()


def main() -> int:
    load_env()
    uri = getenv("NEO4J_URI", required=True)
    user = getenv("NEO4J_USER", required=True)
    password = getenv("NEO4J_PASSWORD", required=True)
    database = getenv("NEO4J_DATABASE", None)
    output_dir = Path(getenv("MIGRATION_OUTPUT_DIR", "graph-migration/output")).resolve()
    neo4j_dir = output_dir / "neo4j_csv"

    manifest_path = neo4j_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing manifest.json. Run transform_to_neo4j.py first.")

    schema_path = neo4j_dir / "neo4j_schema.cypher"
    import_path = neo4j_dir / "neo4j_import.cypher"
    if not schema_path.exists() or not import_path.exists():
        raise SystemExit("Missing cypher files. Run transform_to_neo4j.py first.")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database=database) as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        allow_nonempty = getenv("NEO4J_ALLOW_NONEMPTY", "false").lower() in ("1", "true", "yes")
        clear_db = getenv("NEO4J_CLEAR_DATABASE", "false").lower() in ("1", "true", "yes")

        if node_count > 0 and clear_db:
            log("Target Neo4j database is not empty. Clearing all nodes and relationships.")
            session.run("MATCH (n) DETACH DELETE n").consume()
        elif node_count > 0 and not allow_nonempty:
            raise SystemExit(
                "Target Neo4j database is not empty. Set NEO4J_ALLOW_NONEMPTY=true "
                "or NEO4J_CLEAR_DATABASE=true to proceed."
            )

        # Ensure APOC is available
        try:
            session.run("RETURN apoc.version() AS v").single()
        except Exception as exc:
            raise SystemExit(
                "APOC is required for JSON property import. Enable APOC in Neo4j and retry."
            ) from exc

        log("Creating constraints and indexes...")
        run_cypher_file(session, schema_path)

        log("Importing nodes and relationships...")
        run_cypher_file(session, import_path)

    driver.close()
    log("Import completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
