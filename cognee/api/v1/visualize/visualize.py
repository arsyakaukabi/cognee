import os
import asyncio

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR
from pathlib import Path

logger = get_logger()


async def visualize_graph(
    destination_file_path: str = None,
    node_limit: int | None = None,
    order_by_newest: bool = True,
):
    was_provided = destination_file_path is not None
    if not was_provided:
        repo_root = Path(__file__).resolve().parents[3]
        destination_dir = repo_root / ".data_storage" / "visualizations"
        destination_file_path = str(destination_dir / "graph_visualization.html")

    graph_engine = await get_graph_engine()
    # Default limit from env if not explicitly set
    if node_limit is None:
        env_node_limit = os.getenv("VISUALIZATION_NODE_LIMIT")
        if env_node_limit and env_node_limit.isdigit():
            node_limit = int(env_node_limit)

    graph_data = await graph_engine.get_graph_data(limit=node_limit, order_by_newest=order_by_newest)

    graph = await cognee_network_visualization(graph_data, destination_file_path)

    if was_provided:
        logger.info(f"The HTML file has been stored at path: {destination_file_path}")
    else:
        logger.info(
            "The HTML file has been stored inside the repository at "
            f"{destination_file_path}"
        )

    return graph


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(visualize_graph())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
