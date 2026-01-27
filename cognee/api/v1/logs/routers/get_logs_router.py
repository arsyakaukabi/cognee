import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from cognee.shared.log_stream import register_listener, remove_listener


def get_logs_router() -> APIRouter:
    router = APIRouter()

    @router.get("/stream")
    async def stream_logs():
        """
        Server-Sent Events (SSE) endpoint to stream application logs to the frontend.
        """

        async def event_generator():
            queue = await register_listener()
            try:
                while True:
                    msg = await queue.get()
                    yield f"data: {msg}\n\n"
            except asyncio.CancelledError:
                # Client disconnected
                pass
            finally:
                await remove_listener(queue)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return router
