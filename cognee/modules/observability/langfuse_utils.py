import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from cognee.modules.observability.get_observe import get_langfuse_context
from cognee.shared.logging_utils import resolve_logs_dir

_LAST_FLUSH_ATTEMPT = 0.0
_FLUSH_INTERVAL_SECONDS = 60
_QUEUE_FILE_NAME = "langfuse_offline_queue.jsonl"


def _get_queue_path() -> Path:
    logs_dir = resolve_logs_dir()
    if logs_dir is None:
        return Path.cwd() / _QUEUE_FILE_NAME
    return logs_dir / _QUEUE_FILE_NAME


def _enqueue_langfuse_update(payload: Dict[str, Any]) -> None:
    try:
        payload["queued_at"] = datetime.now(timezone.utc).isoformat()
        queue_path = _get_queue_path()
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        return


def _send_queued_payload(langfuse_client, payload: Dict[str, Any]) -> None:
    metadata = payload.get("metadata")
    queued_at = payload.get("queued_at")
    trace_metadata = {"offline_replay": True, "queued_at": queued_at}
    if isinstance(metadata, dict):
        trace_metadata.update(metadata)

    timestamp = payload.get("timestamp")
    try:
        timestamp_dt = datetime.fromisoformat(timestamp) if timestamp else None
    except Exception:
        timestamp_dt = None

    trace = langfuse_client.trace(
        name="offline-replay",
        input=payload.get("input"),
        output=payload.get("output"),
        metadata=trace_metadata,
        timestamp=timestamp_dt,
    )
    trace.generation(
        name="offline-replay",
        model=payload.get("model"),
        input=payload.get("input"),
        output=payload.get("output"),
        usage_details=payload.get("usage"),
        metadata=payload.get("metadata"),
        start_time=timestamp_dt,
        end_time=timestamp_dt,
    )


def _flush_langfuse_queue() -> None:
    queue_path = _get_queue_path()
    if not queue_path.exists():
        return

    try:
        langfuse_module = importlib.import_module("langfuse")
        langfuse_client = langfuse_module.Langfuse()
    except Exception:
        return

    remaining_lines = []
    try:
        with queue_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    _send_queued_payload(langfuse_client, payload)
                except Exception:
                    remaining_lines.append(line)
    except Exception:
        return

    try:
        if remaining_lines:
            queue_path.write_text("".join(remaining_lines), encoding="utf-8")
        else:
            queue_path.unlink(missing_ok=True)
    except Exception:
        return


def _try_flush_queue() -> None:
    global _LAST_FLUSH_ATTEMPT
    now = time.time()
    if now - _LAST_FLUSH_ATTEMPT < _FLUSH_INTERVAL_SECONDS:
        return
    _LAST_FLUSH_ATTEMPT = now
    _flush_langfuse_queue()


def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
    if response is None:
        return None

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        raw = getattr(response, "_raw_response", None) or getattr(response, "_response", None)
        if isinstance(raw, dict):
            usage = raw.get("usage")
        else:
            usage = getattr(raw, "usage", None)

    return _normalize_usage(usage)


def _normalize_usage(usage: Any) -> Optional[Dict[str, int]]:
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(
            usage, "output_tokens", None
        )
        total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def update_langfuse_observation(
    *,
    input: Any = None,
    output: Any = None,
    model: Optional[str] = None,
    usage: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    langfuse_context = get_langfuse_context()
    if not langfuse_context:
        return

    update_kwargs: Dict[str, Any] = {}
    if input is not None:
        update_kwargs["input"] = input
    if output is not None:
        update_kwargs["output"] = output
    if model is not None:
        update_kwargs["model"] = model
    if metadata is not None:
        update_kwargs["metadata"] = metadata

    normalized_usage = _normalize_usage(usage)
    if normalized_usage:
        update_kwargs["usage"] = normalized_usage

    if not update_kwargs:
        return

    try:
        _try_flush_queue()
        langfuse_context.update_current_observation(**update_kwargs)
    except Exception:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": update_kwargs.get("input"),
            "output": update_kwargs.get("output"),
            "model": update_kwargs.get("model"),
            "usage": update_kwargs.get("usage"),
            "metadata": update_kwargs.get("metadata"),
        }
        _enqueue_langfuse_update(payload)
        return


def update_langfuse_observation_from_response(
    *,
    response: Any,
    input: Any = None,
    output: Any = None,
    model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    usage = _extract_usage(response)
    update_langfuse_observation(
        input=input,
        output=output,
        model=model,
        usage=usage,
        metadata=metadata,
    )


def extract_usage_from_response(response: Any) -> Optional[Dict[str, int]]:
    return _extract_usage(response)
