from pathlib import Path
from kuzu import Connection
from kuzu.database import Database

def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env

def resolve_kuzu_db_path(repo_root: Path) -> Path:
    env = load_env(repo_root / ".env")
    provider = env.get("GRAPH_DATABASE_PROVIDER", "kuzu").lower()
    graph_file_path = env.get("GRAPH_FILE_PATH")
    graph_filename = env.get("GRAPH_FILENAME") or f"cognee_graph_{provider}"
    if graph_file_path:
        return Path(graph_file_path) / graph_filename
    # default path: repo/cognee/.cognee_system/databases/<filename>
    return repo_root / "cognee" / ".cognee_system" / "databases" / graph_filename

repo_root = Path(__file__).resolve().parent
db_path = resolve_kuzu_db_path(repo_root)
print(f"db_path: {db_path}")

env = load_env(repo_root / ".env")
read_only = env.get("GRAPH_DB_READ_ONLY", "true").lower() in {"1", "true", "yes", "y"}
db = Database(str(db_path), read_only=read_only)
conn = Connection(db)

def run(query: str):
    print(f"\nQUERY: {query}")
    result = conn.execute(query)
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    for row in rows:
        print(row)
    print(f"rows: {len(rows)}")

run("MATCH (n) RETURN n LIMIT 5;")
run("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 5;")
run("MATCH (n) RETURN COUNT(n);")
