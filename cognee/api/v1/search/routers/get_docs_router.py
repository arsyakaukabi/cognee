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
    id_knowledge: str = Field(description="ID of the knowledge/document to retrieve details for (UUID)")


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
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from sqlalchemy import select
    from uuid import UUID
    
    # Look up doc_name by id_knowledge
    doc_name = None
    try:
        # Try to parse as UUID
        doc_id = UUID(payload.id_knowledge)
        
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            stmt = select(Data.name).where(Data.id == doc_id)
            result = await session.execute(stmt)
            row = result.first()
            if row and row[0]:
                doc_name = row[0]
    except (ValueError, Exception):
        # If not a valid UUID or DB error, fallback to using id_knowledge as name
        doc_name = payload.id_knowledge
    
    if not doc_name:
        # Fallback to using id_knowledge directly as doc_name
        doc_name = payload.id_knowledge
    
    details = await get_document_details(doc_name)
    return DocsResponseDTO(
        filename=details.get("filename"),
        file_size=details.get("file_size"),
        created_at=details.get("created_at"),
        updated_at=details.get("updated_at"),
        summaries=details.get("summaries", [])
    )
