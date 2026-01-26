#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    raise SystemExit("neo4j package is required. Install with: pip install -r graph-migration/requirements.txt") from exc

from lib import load_env, getenv, log, cypher_escape


def main() -> int:
    load_env()
    uri = getenv("NEO4J_URI", required=True)
    user = getenv("NEO4J_USER", required=True)
    password = getenv("NEO4J_PASSWORD", required=True)
    database = getenv("NEO4J_DATABASE", None)
    output_dir = Path(getenv("MIGRATION_OUTPUT_DIR", "graph-migration/output")).resolve()
    neo4j_dir = output_dir / "neo4j_csv"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = neo4j_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing manifest.json. Run transform_to_neo4j.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    driver = GraphDatabase.driver(uri, auth=(user, password))
    report = {
        "node_total_expected": manifest["nodes"]["total"],
        "edge_total_expected": manifest["relationships"]["total"],
        "node_type_expected": {k: v["count"] for k, v in manifest["nodes"]["types"].items()},
        "relationship_type_expected": {
            k: v["count"] for k, v in manifest["relationships"]["types"].items()
        },
    }

    with driver.session(database=database) as session:
        report["node_total_actual"] = session.run(
            "MATCH (n:`__Node__`) RETURN count(n) AS c"
        ).single()["c"]
        report["edge_total_actual"] = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()[
            "c"
        ]

        # Per-label counts
        node_type_actual: Dict[str, int] = {}
        for label in report["node_type_expected"].keys():
            label_token = cypher_escape(label)
            result = session.run(
                f"MATCH (n:`__Node__`:{label_token}) RETURN count(n) AS c"
            ).single()["c"]
            node_type_actual[label] = result
        report["node_type_actual"] = node_type_actual

        # Per-relationship counts
        rel_type_actual: Dict[str, int] = {}
        for rel in report["relationship_type_expected"].keys():
            rel_token = cypher_escape(rel)
            result = session.run(f"MATCH ()-[r:{rel_token}]->() RETURN count(r) AS c").single()[
                "c"
            ]
            rel_type_actual[rel] = result
        report["relationship_type_actual"] = rel_type_actual

        # Uniqueness check
        duplicates = session.run(
            "MATCH (n:`__Node__`) WITH n.id AS id, count(*) AS c WHERE c > 1 RETURN count(*) AS dup"
        ).single()["dup"]
        report["duplicate_node_ids"] = duplicates

        # Relationship property presence check
        missing_rel_name = session.run(
            "MATCH ()-[r]->() WHERE r.relationship_name IS NULL RETURN count(r) AS c"
        ).single()["c"]
        report["relationships_missing_relationship_name"] = missing_rel_name

    driver.close()

    # Compute diffs
    report["node_total_match"] = report["node_total_expected"] == report["node_total_actual"]
    report["edge_total_match"] = report["edge_total_expected"] == report["edge_total_actual"]
    report["node_type_mismatches"] = {
        k: {
            "expected": report["node_type_expected"][k],
            "actual": report["node_type_actual"][k],
        }
        for k in report["node_type_expected"].keys()
        if report["node_type_expected"][k] != report["node_type_actual"][k]
    }
    report["relationship_type_mismatches"] = {
        k: {
            "expected": report["relationship_type_expected"][k],
            "actual": report["relationship_type_actual"][k],
        }
        for k in report["relationship_type_expected"].keys()
        if report["relationship_type_expected"][k] != report["relationship_type_actual"][k]
    }

    json_report = report_dir / "validation_report.json"
    json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text_lines = [
        f"node_total_expected={report['node_total_expected']}",
        f"node_total_actual={report['node_total_actual']}",
        f"edge_total_expected={report['edge_total_expected']}",
        f"edge_total_actual={report['edge_total_actual']}",
        f"duplicate_node_ids={report['duplicate_node_ids']}",
        f"relationships_missing_relationship_name={report['relationships_missing_relationship_name']}",
        f"node_total_match={report['node_total_match']}",
        f"edge_total_match={report['edge_total_match']}",
        f"node_type_mismatches={len(report['node_type_mismatches'])}",
        f"relationship_type_mismatches={len(report['relationship_type_mismatches'])}",
    ]
    (report_dir / "validation_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    log(f"Validation report written to {json_report}")

    # Decide exit code
    if (
        not report["node_total_match"]
        or not report["edge_total_match"]
        or report["duplicate_node_ids"] > 0
        or report["relationships_missing_relationship_name"] > 0
        or report["node_type_mismatches"]
        or report["relationship_type_mismatches"]
    ):
        log("Validation failed. See report for details.")
        return 2

    log("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
