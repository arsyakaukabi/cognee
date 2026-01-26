import base64
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_env(env_path: Optional[Path] = None) -> None:
    """Load KEY=VALUE lines from a .env file without external deps."""
    path = env_path or (Path(__file__).resolve().parent.parent / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def getenv(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise SystemExit(f"Missing required env var: {name}")
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def safe_filename(value: str) -> str:
    value = value.strip() or "unknown"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    if not value:
        return "unknown"
    return value


def cypher_escape(name: str) -> str:
    """Escape a label or relationship type for Cypher by wrapping in backticks."""
    name = name.replace("`", "``")
    return f"`{name}`"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def b64_encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def try_parse_json_string(value: str) -> Optional[Any]:
    stripped = value.strip()
    if not stripped:
        return None
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def parse_json_map(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    raw_str = str(raw).strip()
    if raw_str == "":
        return {}
    try:
        data = json.loads(raw_str)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}


def normalize_timestamp(value: Any) -> str:
    """Return ISO-8601 string in UTC, or empty string if missing."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        # Heuristic for epoch seconds/millis/micros
        v = float(value)
        if v > 1e14:
            dt = datetime.fromtimestamp(v / 1e6, tz=timezone.utc)
        elif v > 1e11:
            dt = datetime.fromtimestamp(v / 1e3, tz=timezone.utc)
        elif v > 1e9:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
        else:
            return ""
        return dt.isoformat()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        # Normalize trailing Z to +00:00 for datetime()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            return value
    return str(value)


def open_csv_writer(path: Path, fieldnames: Iterable[str], mode: str = "w") -> csv.DictWriter:
    ensure_dir(path.parent)
    write_header = (mode == "w") or (mode == "a" and not path.exists()) or (mode == "a" and path.stat().st_size == 0)
    f = path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    # Attach file handle for later close
    writer._file_handle = f  # type: ignore[attr-defined]
    return writer


def close_writer(writer: csv.DictWriter) -> None:
    fh = getattr(writer, "_file_handle", None)
    if fh:
        fh.close()


def flatten_edge_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, str):
            parsed = try_parse_json_string(value)
            if parsed is not None:
                value = parsed
        if key == "weights" and isinstance(value, dict):
            for weight_name, weight_value in value.items():
                flattened[f"weight_{weight_name}"] = weight_value
        elif isinstance(value, dict):
            flattened[f"{key}_json"] = json_dumps(value)
        elif isinstance(value, list):
            flattened[f"{key}_json"] = json_dumps(value)
        else:
            flattened[key] = value
    return flattened


def normalize_node_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, str):
            parsed = try_parse_json_string(value)
            if parsed is not None:
                value = parsed
        normalized[key] = value
    return normalized
