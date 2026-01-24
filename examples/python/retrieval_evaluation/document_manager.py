"""
Document manager for adding and cognifying documents.
"""

from typing import List, Optional
import cognee
from cognee.shared.logging_utils import get_logger

logger = get_logger("DocumentManager")


async def add_documents(file_paths: List[str]) -> None:
    """
    Add documents to Cognee.
    
    Args:
        file_paths: List of file paths to add
    """
    if not file_paths:
        logger.warning("No file paths provided")
        return
    
    logger.info(f"Adding {len(file_paths)} documents to Cognee...")
    await cognee.add(file_paths)
    logger.info("Documents added successfully")


async def cognify_documents() -> None:
    """
    Cognify documents (build knowledge graph).
    """
    logger.info("Cognifying documents...")
    await cognee.cognify()
    logger.info("Documents cognified successfully")


async def verify_graph_ready() -> bool:
    """
    Verify that graph is ready for evaluation.
    
    Returns:
        True if graph is ready, False otherwise
    """
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        
        # Check if we have at least some nodes
        if len(nodes) > 0:
            logger.info(f"Graph is ready: {len(nodes)} nodes, {len(edges)} edges")
            return True
        else:
            logger.warning("Graph is empty")
            return False
    except Exception as e:
        logger.error(f"Error verifying graph: {e}")
        return False
