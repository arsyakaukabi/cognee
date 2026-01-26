import json
from typing import List, Optional, Union, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("api.pipeline")


def get_pipeline_router() -> APIRouter:
    router = APIRouter()
    
    def _event(name: str, payload: dict) -> str:
        """Format an SSE event chunk."""
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"
    
    @router.post("/ingest-and-cognify")
    async def ingest_and_cognify(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        render_scale: Optional[float] = Form(default=None, description="Override PDF render scale"),
        batch_size: Optional[int] = Form(default=None, description="Override PDF page batch size"),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Ingest data then immediately cognify it (no streaming). Returns final statuses.
        """
        add_kwargs = {}
        if render_scale is not None:
            add_kwargs["preferred_loaders"] = {"vision_pdf_loader": {"render_scale": render_scale}}
            if batch_size is not None:
                add_kwargs["preferred_loaders"]["vision_pdf_loader"]["batch_size"] = batch_size
        elif batch_size is not None:
            add_kwargs["preferred_loaders"] = {"vision_pdf_loader": {"batch_size": batch_size}}

        from cognee.api.v1.add import add as cognee_add
        add_run = await cognee_add(
            data,
            datasetName,
            user=user,
            dataset_id=datasetId,
            node_set=node_set if node_set != [""] else None,
            **add_kwargs,
        )

        if isinstance(add_run, PipelineRunErrored):
            return JSONResponse(
                status_code=420,
                content={
                    "status": "add_failed",
                    "message": "Ingestion failed.",
                    "details": add_run.model_dump(),
                },
            )

        dataset_uuid = add_run.dataset_id if hasattr(add_run, "dataset_id") else None
        if dataset_uuid is None and datasetId:
            dataset_uuid = datasetId
        if dataset_uuid is None:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "add_failed",
                    "message": "Ingestion finished but dataset ID was not returned.",
                },
            )

        from cognee.api.v1.cognify import cognify as cognee_cognify
        cognify_run = await cognee_cognify(
            datasets=[dataset_uuid],
            user=user,
            run_in_background=False,
        )

        run_info = None
        if isinstance(cognify_run, dict):
            run_info = cognify_run.get(dataset_uuid)

        if run_info is None:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "cognify_failed",
                    "message": "Cognify did not return run info.",
                },
            )

        return {
            "status": "completed",
            "datasetId": str(dataset_uuid),
            "addPipelineRunId": str(add_run.pipeline_run_id)
            if hasattr(add_run, "pipeline_run_id")
            else None,
            "cognifyPipelineRunId": str(run_info.pipeline_run_id)
            if hasattr(run_info, "pipeline_run_id")
            else None,
        }

    @router.post("/ingest-and-cognify/stream", response_class=StreamingResponse)
    async def ingest_and_cognify_stream(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        render_scale: Optional[float] = Form(default=None, description="Override PDF render scale"),
        batch_size: Optional[int] = Form(default=None, description="Override PDF page batch size"),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Ingest data and immediately cognify it, streaming user-friendly progress events (SSE).

        Events:
        - accepted: upload received
        - add_started / add_completed / add_failed
        - cognify_started / cognify_completed / cognify_failed
        - done: both steps succeeded
        """

        async def _stream():
            yield _event(
                "accepted",
                {
                    "message": "Upload received. Starting ingestion.",
                    "datasetName": datasetName,
                    "datasetId": str(datasetId) if datasetId else None,
                },
            )

            add_kwargs = {}
            if render_scale is not None:
                add_kwargs["preferred_loaders"] = {"vision_pdf_loader": {"render_scale": render_scale}}
                if batch_size is not None:
                    add_kwargs["preferred_loaders"]["vision_pdf_loader"]["batch_size"] = batch_size
            elif batch_size is not None:
                add_kwargs["preferred_loaders"] = {"vision_pdf_loader": {"batch_size": batch_size}}
            # Ingest
            try:
                yield _event("add_started", {"message": "Ingestion started."})
                from cognee.api.v1.add import add as cognee_add

                add_run = await cognee_add(
                    data,
                    datasetName,
                    user=user,
                    dataset_id=datasetId,
                    node_set=node_set if node_set != [""] else None,
                    **add_kwargs,
                )
                if isinstance(add_run, PipelineRunErrored):
                    yield _event(
                        "add_failed",
                        {"message": "Ingestion failed.", "details": add_run.model_dump()},
                    )
                    return

                dataset_uuid = (
                    add_run.dataset_id if hasattr(add_run, "dataset_id") else None
                )
                if dataset_uuid is None and datasetId:
                    dataset_uuid = datasetId
                if dataset_uuid is None:
                    yield _event(
                        "add_failed",
                        {
                            "message": "Ingestion finished but dataset ID was not returned.",
                            "details": "Cannot continue to cognify without a dataset.",
                        },
                    )
                    return

                yield _event(
                    "add_completed",
                    {
                        "message": "Ingestion finished.",
                        "datasetId": str(dataset_uuid),
                        "pipelineRunId": str(add_run.pipeline_run_id)
                        if hasattr(add_run, "pipeline_run_id")
                        else None,
                    },
                )
            except Exception as error:
                yield _event(
                    "add_failed",
                    {"message": "Ingestion failed.", "details": str(error)},
                )
                return

            # Cognify
            try:
                yield _event(
                    "cognify_started",
                    {"message": "Building knowledge graph (cognify) has started."},
                )
                from cognee.api.v1.cognify import cognify as cognee_cognify

                cognify_run = await cognee_cognify(
                    datasets=[dataset_uuid],
                    user=user,
                    run_in_background=False,
                )

                run_info = None
                if isinstance(cognify_run, dict):
                    run_info = cognify_run.get(dataset_uuid)

                yield _event(
                    "cognify_completed",
                    {
                        "message": "Knowledge graph built.",
                        "datasetId": str(dataset_uuid),
                        "pipelineRunId": str(run_info.pipeline_run_id)
                        if run_info and hasattr(run_info, "pipeline_run_id")
                        else None,
                    },
                )
            except Exception as error:
                yield _event(
                    "cognify_failed",
                    {"message": "Cognify failed.", "details": str(error)},
                )
                return

            yield _event(
                "done",
                {
                    "message": "Ingestion and cognify completed successfully.",
                    "datasetId": str(dataset_uuid),
                },
            )

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return router
