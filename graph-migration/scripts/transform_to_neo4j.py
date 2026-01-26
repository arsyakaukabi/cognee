#!/usr/bin/env python3
import csv
import json
import hashlib
from pathlib import Path
import sys
from typing import Dict

from lib import (
    load_env,
    getenv,
    ensure_dir,
    log,
    parse_json_map,
    normalize_timestamp,
    safe_filename,
    cypher_escape,
    json_dumps,
    b64_encode,
    flatten_edge_properties,
    normalize_node_properties,
    open_csv_writer,
    close_writer,
)


class WriterManager:
    def __init__(self, base_dir: Path, fieldnames, max_writers: int = 200):
        self.base_dir = base_dir
        self.fieldnames = fieldnames
        self.max_writers = max_writers
        self.writers: Dict[str, csv.DictWriter] = {}
        # Keep track of paths we've ever opened so we know whether to append
        self.known_paths: Dict[str, Path] = {}

    def get(self, key: str, safe_name: str) -> csv.DictWriter:
        # If cache hit, move to end (MRU)
        if key in self.writers:
            # Python dicts are ordered insertion-preservation in 3.7+
            # Re-inserting to update order
            writer = self.writers.pop(key)
            self.writers[key] = writer
            return writer

        # If cache full, pop first item (LRU)
        if len(self.writers) >= self.max_writers:
            # pop first key (LRU)
            _k, _w = next(iter(self.writers.items()))
            self.writers.pop(_k)
            close_writer(_w)

        path = self.base_dir / f"{safe_name}.csv"
        # If we saw this key before, append. If not, overwrite (first time).
        mode = "a" if key in self.known_paths else "w"
        
        writer = open_csv_writer(path, self.fieldnames, mode=mode)
        self.writers[key] = writer
        self.known_paths[key] = path
        return writer

    def close_all(self) -> None:
        for writer in self.writers.values():
            close_writer(writer)
        self.writers.clear()


def _safe_name_with_collision(label: str, used: Dict[str, str]) -> str:
    safe = safe_filename(label)
    if safe in used and used[safe] != label:
        suffix = hashlib.md5(label.encode("utf-8")).hexdigest()[:6]
        safe = f"{safe}_{suffix}"
    used[safe] = label
    return safe


def transform_nodes(nodes_csv: Path, out_dir: Path):
    node_dir = out_dir / "nodes"
    ensure_dir(node_dir)
    manager = WriterManager(
        node_dir,
        ["id", "name", "type", "created_at", "updated_at", "properties_b64"],
    )

    counts: Dict[str, int] = {}
    file_map: Dict[str, str] = {}
    safe_used: Dict[str, str] = {}
    total = 0
    invalid = 0

    with nodes_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = (row.get("id") or "").strip()
            if not node_id:
                invalid += 1
                continue
            raw_type = (row.get("type") or "").strip() or "Unknown"
            safe = _safe_name_with_collision(raw_type, safe_used)
            writer = manager.get(raw_type, safe)
            file_map[raw_type] = f"nodes/{safe}.csv"

            raw_props = parse_json_map(row.get("properties"))
            # remove core fields if they leak into properties
            for key in ("id", "name", "type", "created_at", "updated_at"):
                raw_props.pop(key, None)

            props = normalize_node_properties(raw_props)
            props_json = json_dumps(props if props else {})
            props_b64 = b64_encode(props_json)

            writer.writerow(
                {
                    "id": node_id,
                    "name": row.get("name") or "",
                    "type": raw_type,
                    "created_at": normalize_timestamp(row.get("created_at")),
                    "updated_at": normalize_timestamp(row.get("updated_at")),
                    "properties_b64": props_b64,
                }
            )
            counts[raw_type] = counts.get(raw_type, 0) + 1
            total += 1

    manager.close_all()
    return {
        "counts": counts,
        "files": file_map,
        "total": total,
        "invalid": invalid,
    }


def transform_edges(edges_csv: Path, out_dir: Path):
    edge_dir = out_dir / "edges"
    ensure_dir(edge_dir)
    manager = WriterManager(
        edge_dir,
        [
            "from_id",
            "to_id",
            "relationship_name",
            "created_at",
            "updated_at",
            "properties_b64",
        ],
    )

    counts: Dict[str, int] = {}
    file_map: Dict[str, str] = {}
    safe_used: Dict[str, str] = {}
    total = 0
    invalid = 0

    with edges_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = (row.get("from_id") or "").strip()
            to_id = (row.get("to_id") or "").strip()
            rel_name = (row.get("relationship_name") or "").strip() or "UNKNOWN_REL"

            if not from_id or not to_id:
                invalid += 1
                continue

            safe = _safe_name_with_collision(rel_name, safe_used)
            writer = manager.get(rel_name, safe)
            file_map[rel_name] = f"edges/{safe}.csv"

            raw_props = parse_json_map(row.get("properties"))
            # ensure relationship_name and endpoints are present
            raw_props.setdefault("relationship_name", rel_name)
            raw_props.setdefault("source_node_id", from_id)
            raw_props.setdefault("target_node_id", to_id)

            props = flatten_edge_properties(raw_props)
            props_json = json_dumps(props if props else {})
            props_b64 = b64_encode(props_json)

            writer.writerow(
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "relationship_name": rel_name,
                    "created_at": normalize_timestamp(row.get("created_at")),
                    "updated_at": normalize_timestamp(row.get("updated_at")),
                    "properties_b64": props_b64,
                }
            )
            counts[rel_name] = counts.get(rel_name, 0) + 1
            total += 1

    manager.close_all()
    return {
        "counts": counts,
        "files": file_map,
        "total": total,
        "invalid": invalid,
    }


def generate_schema_cypher(out_dir: Path) -> Path:
    schema_path = out_dir / "neo4j_schema.cypher"
    schema_path.write_text(
        "\n".join(
            [
                "// Base constraint required by Neo4j adapter",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`__Node__`) REQUIRE n.id IS UNIQUE;",
                "// Helpful indexes for common filters",
                "CREATE INDEX IF NOT EXISTS FOR (n:`__Node__`) ON (n.type);",
                "CREATE INDEX IF NOT EXISTS FOR (n:`__Node__`) ON (n.name);",
                "// Relationship property indexes require a specific relationship type in Neo4j.",
                "// If needed, add per-type indexes later after import.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return schema_path


def generate_import_cypher(manifest: dict, out_dir: Path, batch_size: int) -> Path:
    import_path = out_dir / "neo4j_import.cypher"
    lines = []

    for label, info in sorted(manifest["nodes"]["types"].items()):
        file_uri = f"file:///{info['file']}"
        label_token = cypher_escape(label)
        lines.append(f"// Nodes: {label}")
        lines.append(
            "\n".join(
                [
                    "CALL {",
                    f"  LOAD CSV WITH HEADERS FROM '{file_uri}' AS row",
                    f"  MERGE (n:`__Node__`:{label_token} {{id: row.id}})",
                    "  SET n.name = row.name",
                    "  SET n.type = row.type",
                    "  SET n.created_at = CASE row.created_at WHEN '' THEN NULL ELSE datetime(row.created_at) END",
                    "  SET n.updated_at = CASE row.updated_at WHEN '' THEN NULL ELSE datetime(row.updated_at) END",
                    "  WITH n, apoc.convert.fromJsonMap(apoc.text.base64Decode(row.properties_b64)) AS props",
                    "  WITH n, apoc.map.fromPairs([k IN keys(props) | [k,",
                    "    CASE",
                    "      WHEN props[k] IS NULL THEN NULL",
                    "      WHEN apoc.meta.cypher.type(props[k]) = 'MAP' THEN apoc.convert.toJson(props[k])",
                    "      ELSE props[k]",
                    "    END",
                    "  ]]) AS flat_props",
                    "  SET n += flat_props",
                    f"}} IN TRANSACTIONS OF {batch_size} ROWS",
                    "FINISH;",
                    "",
                ]
            )
        )

    for rel, info in sorted(manifest["relationships"]["types"].items()):
        file_uri = f"file:///{info['file']}"
        rel_token = cypher_escape(rel)
        lines.append(f"// Relationships: {rel}")
        lines.append(
            "\n".join(
                [
                    "CALL {",
                    f"  LOAD CSV WITH HEADERS FROM '{file_uri}' AS row",
                    "  MATCH (a:`__Node__` {id: row.from_id})",
                    "  MATCH (b:`__Node__` {id: row.to_id})",
                    f"  MERGE (a)-[r:{rel_token}]->(b)",
                    "  SET r.relationship_name = row.relationship_name",
                    "  SET r.created_at = CASE row.created_at WHEN '' THEN NULL ELSE datetime(row.created_at) END",
                    "  SET r.updated_at = CASE row.updated_at WHEN '' THEN NULL ELSE datetime(row.updated_at) END",
                    "  WITH r, apoc.convert.fromJsonMap(apoc.text.base64Decode(row.properties_b64)) AS props",
                    "  WITH r, apoc.map.fromPairs([k IN keys(props) | [k,",
                    "    CASE",
                    "      WHEN props[k] IS NULL THEN NULL",
                    "      WHEN apoc.meta.cypher.type(props[k]) = 'MAP' THEN apoc.convert.toJson(props[k])",
                    "      ELSE props[k]",
                    "    END",
                    "  ]]) AS flat_props",
                    "  SET r += flat_props",
                    f"}} IN TRANSACTIONS OF {batch_size} ROWS",
                    "FINISH;",
                    "",
                ]
            )
        )

    import_path.write_text("\n".join(lines), encoding="utf-8")
    return import_path


def main() -> int:
    load_env()
    output_dir = Path(getenv("MIGRATION_OUTPUT_DIR", "graph-migration/output")).resolve()
    raw_dir = output_dir / "raw"
    neo4j_dir = output_dir / "neo4j_csv"
    ensure_dir(neo4j_dir)

    nodes_csv = Path(getenv("KUZU_NODES_CSV", str(raw_dir / "nodes.csv"))).resolve()
    edges_csv = Path(getenv("KUZU_EDGES_CSV", str(raw_dir / "edges.csv"))).resolve()
    batch_size = int(getenv("NEO4J_BATCH_SIZE", "10000"))

    if not nodes_csv.exists() or not edges_csv.exists():
        raise SystemExit("Missing raw export CSVs. Run export_kuzu.py first.")

    log("Transforming nodes...")
    nodes_info = transform_nodes(nodes_csv, neo4j_dir)
    log("Transforming edges...")
    edges_info = transform_edges(edges_csv, neo4j_dir)

    manifest = {
        "nodes": {
            "types": {
                label: {
                    "count": count,
                    "file": nodes_info["files"][label],
                }
                for label, count in nodes_info["counts"].items()
            },
            "total": nodes_info["total"],
            "invalid": nodes_info["invalid"],
        },
        "relationships": {
            "types": {
                rel: {
                    "count": count,
                    "file": edges_info["files"][rel],
                }
                for rel, count in edges_info["counts"].items()
            },
            "total": edges_info["total"],
            "invalid": edges_info["invalid"],
        },
    }

    schema_path = generate_schema_cypher(neo4j_dir)
    import_path = generate_import_cypher(manifest, neo4j_dir, batch_size)
    manifest["files"] = {
        "neo4j_schema_cypher": str(schema_path),
        "neo4j_import_cypher": str(import_path),
    }

    manifest_path = neo4j_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"Wrote manifest: {manifest_path}")
    log(f"Wrote schema cypher: {schema_path}")
    log(f"Wrote import cypher: {import_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
