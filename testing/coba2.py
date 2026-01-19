# coba.py
from pathlib import Path

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

repo_root = Path(__file__).resolve().parent
env = load_env(repo_root / ".env")

provider = env.get("GRAPH_DATABASE_PROVIDER", "kuzu")
graph_file_path = env.get("GRAPH_FILE_PATH")
graph_filename = env.get("GRAPH_FILENAME") or f"cognee_graph_{provider.lower()}"

if provider.lower() == "kuzu":
    if graph_file_path:
        db_path = Path(graph_file_path) / graph_filename
    else:
        # default path based on cognee/root_dir.py (package dir)
        package_dir = repo_root / "cognee"
        db_path = package_dir / ".cognee_system" / "databases" / graph_filename
    print(f"provider: {provider}")
    print(f"kuzu_db_path: {db_path}")
else:
    print(f"provider: {provider} (non-kuzu)")

