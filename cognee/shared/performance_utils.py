import time
import functools
import inspect
import asyncio
from typing import Optional, Any
from contextvars import ContextVar
import structlog
from uuid import uuid4

logger = structlog.get_logger("performance")

# Context Variables for tracing state
request_correlation_id: ContextVar[Optional[str]] = ContextVar("request_correlation_id", default=None)
trace_stack_depth: ContextVar[int] = ContextVar("trace_stack_depth", default=0)
debug_trace_enabled: ContextVar[bool] = ContextVar("debug_trace_enabled", default=False)

def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one if missing."""
    cid = request_correlation_id.get()
    if not cid:
        cid = str(uuid4())
        request_correlation_id.set(cid)
    return cid

def set_debug_trace(enabled: bool):
    """Enable or disable debug tracing for the current context."""
    debug_trace_enabled.set(enabled)

class TraceSpan:
    """
    Context manager to trace a block of code.
    Logs start and end events with duration if debug tracing is enabled.
    """
    def __init__(self, name: str, component: str = "core", **kwargs):
        self.name = name
        self.component = component
        self.kwargs = kwargs
        self.start_time = 0.0

    def __enter__(self):
        if not debug_trace_enabled.get():
            return self
        
        self.start_time = time.perf_counter()
        depth = trace_stack_depth.get()
        trace_stack_depth.set(depth + 1)
        
        # Log start event
        logger.info(
            "span_start",
            span_name=self.name,
            component=self.component,
            depth=depth,
            correlation_id=get_correlation_id(),
            **self.kwargs
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not debug_trace_enabled.get():
            return

        duration_ms = (time.perf_counter() - self.start_time) * 1000
        depth = trace_stack_depth.get() - 1
        trace_stack_depth.set(depth)
        
        # Log end event
        logger.info(
            "span_end",
            span_name=self.name,
            component=self.component,
            duration_ms=round(duration_ms, 3),
            depth=depth,
            correlation_id=get_correlation_id(),
            **self.kwargs
        )

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

def trace_perf(name: Optional[str] = None, component: str = "core"):
    """
    Decorator to measure execution time of a function.
    Usage: @trace_perf(name="custom_name", component="db")
    """
    def decorator(func):
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with TraceSpan(span_name, component):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with TraceSpan(span_name, component):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
