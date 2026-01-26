from typing import List, Optional
from pydantic import Field
from fastapi import APIRouter

from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.search.retrieval import get_document_details

router = APIRouter()


def get_docs_router():
    return router


class DocsPayloadDTO(InDTO):
    """Payload for document details endpoint."""
    name: str = Field(description="Name of the document to retrieve details for")


class DocsResponseDTO(OutDTO):
    """Document details response."""
    filename: Optional[str] = Field(default=None, description="Filename (doc_name + extension)")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    created_at: Optional[int] = Field(default=None, description="Document creation timestamp (epoch ms)")
    updated_at: Optional[int] = Field(default=None, description="Document update timestamp (epoch ms)")
    summaries: List[str] = Field(default=[], description="List of chunk summaries")


@router.post("/", response_model=DocsResponseDTO)
async def get_document_details_endpoint(payload: DocsPayloadDTO):
    """
    Retrieve document details (filename, metadata, and summaries).
    """
    details = await get_document_details(payload.name)
    return DocsResponseDTO(
        filename=details.get("filename"),
        file_size=details.get("file_size"),
        created_at=details.get("created_at"),
        updated_at=details.get("updated_at"),
        summaries=details.get("summaries", [])
    )
