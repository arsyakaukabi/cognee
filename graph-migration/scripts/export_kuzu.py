#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

try:
    import kuzu
except ImportError as exc:
    raise SystemExit("kuzu package is required. Install with: pip install -r graph-migration/requirements.txt") from exc

from lib import load_env, getenv, ensure_dir, log, normalize_timestamp


def _as_py(value):
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def export_nodes(conn, out_path: Path) -> int:
    query = (
        "MATCH (n:Node) "
        "RETURN n.id AS id, n.name AS name, n.type AS type, "
        "n.created_at AS created_at, n.updated_at AS updated_at, n.properties AS properties"
    )
    result = conn.execute(query)
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "type", "created_at", "updated_at", "properties"],
        )
        writer.writeheader()
        count = 0
        while result.has_next():
            row = result.get_next()
            values = [_as_py(v) for v in row]
            writer.writerow(
                {
                    "id": values[0],
                    "name": values[1],
                    "type": values[2],
                    "created_at": normalize_timestamp(values[3]),
                    "updated_at": normalize_timestamp(values[4]),
                    "properties": values[5],
                }
            )
            count += 1
    return count


def export_edges(conn, out_path: Path) -> int:
    query = (
        "MATCH (from:Node)-[r:EDGE]->(to:Node) "
        "RETURN from.id AS from_id, to.id AS to_id, r.relationship_name AS relationship_name, "
        "r.created_at AS created_at, r.updated_at AS updated_at, r.properties AS properties"
    )
    result = conn.execute(query)
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "from_id",
                "to_id",
                "relationship_name",
                "created_at",
                "updated_at",
                "properties",
            ],
        )
        writer.writeheader()
        count = 0
        while result.has_next():
            row = result.get_next()
            values = [_as_py(v) for v in row]
            writer.writerow(
                {
                    "from_id": values[0],
                    "to_id": values[1],
                    "relationship_name": values[2],
                    "created_at": normalize_timestamp(values[3]),
                    "updated_at": normalize_timestamp(values[4]),
                    "properties": values[5],
                }
            )
            count += 1
    return count


def main() -> int:
    load_env()
    kuzu_db_path = getenv("KUZU_DB_PATH", required=True)
    output_dir = Path(getenv("MIGRATION_OUTPUT_DIR", "graph-migration/output")).resolve()
    raw_dir = output_dir / "raw"
    ensure_dir(raw_dir)

    log(f"Exporting from Kuzu database at: {kuzu_db_path}")
    db = kuzu.Database(kuzu_db_path)
    conn = kuzu.Connection(db)

    try:
        conn.execute("LOAD EXTENSION JSON;")
    except Exception:
        pass

    nodes_csv = raw_dir / "nodes.csv"
    edges_csv = raw_dir / "edges.csv"

    node_count = export_nodes(conn, nodes_csv)
    edge_count = export_edges(conn, edges_csv)

    manifest = raw_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"nodes_csv={nodes_csv}",
                f"edges_csv={edges_csv}",
                f"node_count={node_count}",
                f"edge_count={edge_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log(f"Exported {node_count} nodes to {nodes_csv}")
    log(f"Exported {edge_count} edges to {edges_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
