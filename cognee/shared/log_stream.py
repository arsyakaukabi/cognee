import asyncio
import logging
from typing import Set

listeners: Set[asyncio.Queue] = set()
_lock = asyncio.Lock()


def _get_formatter():
    return logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")


async def register_listener() -> asyncio.Queue:
    """
    Create and register a new listener queue for log streaming.
    """
    queue: asyncio.Queue = asyncio.Queue()
    async with _lock:
        listeners.add(queue)
    return queue


async def remove_listener(queue: asyncio.Queue):
    """
    Remove a listener queue.
    """
    async with _lock:
        listeners.discard(queue)


class SSELogHandler(logging.Handler):
    """
    Simple logging handler that broadcasts log records to all registered asyncio queues.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            loop = asyncio.get_event_loop()
            for q in list(listeners):
                loop.call_soon_threadsafe(q.put_nowait, msg)
        except Exception:
            self.handleError(record)


def attach_stream_handler(level=logging.INFO):
    """
    Attach the SSELogHandler to the root logger (idempotent enough for our use).
    """
    handler = SSELogHandler()
    handler.setLevel(level)
    handler.setFormatter(_get_formatter())
    root = logging.getLogger()
    root.addHandler(handler)
    return handler
