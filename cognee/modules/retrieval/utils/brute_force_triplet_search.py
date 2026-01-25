import asyncio
import time
from typing import List, Optional, Type

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

logger = get_logger(level=ERROR)


def format_triplets(edges):
    print("\n\n\n")

    def filter_attributes(obj, attributes):
        """Helper function to filter out non-None properties, including nested dicts."""
        result = {}
        for attr in attributes:
            value = getattr(obj, attr, None)
            if value is not None:
                # If the value is a dict, extract relevant keys from it
                if isinstance(value, dict):
                    nested_values = {
                        k: v for k, v in value.items() if k in attributes and v is not None
                    }
                    result[attr] = nested_values
                else:
                    result[attr] = value
        return result

    triplets = []
    for edge in edges:
        node1 = edge.node1
        node2 = edge.node2
        edge_attributes = edge.attributes
        node1_attributes = node1.attributes
        node2_attributes = node2.attributes

        # Filter only non-None properties
        node1_info = {key: value for key, value in node1_attributes.items() if value is not None}
        node2_info = {key: value for key, value in node2_attributes.items() if value is not None}
        edge_info = {key: value for key, value in edge_attributes.items() if value is not None}

        # Create the formatted triplet
        triplet = f"Node1: {node1_info}\nEdge: {edge_info}\nNode2: {node2_info}\n\n\n"
        triplets.append(triplet)

    return "".join(triplets)


async def get_memory_fragment(
    properties_to_project: Optional[List[str]] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    relevant_ids_to_filter: Optional[List[str]] = None,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> CogneeGraph:
    """Creates and initializes a CogneeGraph memory fragment with optional property projections."""
    if properties_to_project is None:
        properties_to_project = ["id", "description", "name", "type", "text"]

    memory_fragment = CogneeGraph()

    try:
        graph_engine = await get_graph_engine()

        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=properties_to_project,
            edge_properties_to_project=["relationship_name", "edge_text"],
            node_type=node_type,
            node_name=node_name,
            relevant_ids_to_filter=relevant_ids_to_filter,
            triplet_distance_penalty=triplet_distance_penalty,
        )

    except EntityNotFoundError:
        # This is expected behavior - continue with empty fragment
        pass
    except Exception as e:
        logger.error(f"Error during memory fragment creation: {str(e)}")
        # Still return the fragment even if projection failed
        pass

    return memory_fragment


async def brute_force_triplet_search(
    query: str,
    top_k: int = 5,
    collections: Optional[List[str]] = None,
    properties_to_project: Optional[List[str]] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    wide_search_top_k: Optional[int] = 50,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> List[Edge]:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query.
        properties_to_project (Optional[List[str]]): List of properties to project.
        memory_fragment (Optional[CogneeGraph]): Existing memory fragment to reuse.
        node_type: node type to filter
        node_name: node name to filter
        wide_search_top_k (Optional[int]): Number of initial elements to retrieve from collections
        triplet_distance_penalty (Optional[float]): Default distance penalty in graph projection

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Setting wide search limit based on the parameters
    non_global_search = node_name is None

    wide_search_limit = wide_search_top_k if non_global_search else None

    if collections is None:
        collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
        ]

    if "EdgeType_relationship_name" not in collections:
        collections.append("EdgeType_relationship_name")

    try:
        vector_engine = get_vector_engine()
    except Exception as e:
        logger.error("Failed to initialize vector engine: %s", e)
        raise RuntimeError("Initialization error") from e

    # Filter to collections that actually exist to avoid wasted queries/errors.
    # This does not change quality because missing collections already return empty results.
    # if collections:
    #     exists_flags = await asyncio.gather(
    #         *[vector_engine.has_collection(collection_name) for collection_name in collections]
    #     )
    #     collections = [c for c, ok in zip(collections, exists_flags) if ok]

    # ========== STEP 0: Embed Query ==========
    step0_start = time.time()
    query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]
    step0_duration = (time.time() - step0_start) * 1000
    logger.info(
        f"⏱️ [STEP 0] Embed Query: {step0_duration:.2f}ms"
    )

    async def search_in_collection(collection_name: str):
        try:
            return await vector_engine.search(
                collection_name=collection_name, query_vector=query_vector, limit=wide_search_limit
            )
        except CollectionNotFoundError:
            return []

    try:
        # ========== STEP 1: Vector Search ==========
        step1_start = time.time()
        
        results = await asyncio.gather(
            *[search_in_collection(collection_name) for collection_name in collections]
        )

        if all(not item for item in results):
            return []

        step1_duration = (time.time() - step1_start) * 1000
        logger.info(
            f"⏱️ [STEP 1] Vector Search: {step1_duration:.2f}ms (6 collections parallel)"
        )

        # ========== STEP 2: Build node_distances dict ==========
        step2_start = time.time()
        
        node_distances = {collection: result for collection, result in zip(collections, results)}
        edge_distances = node_distances.get("EdgeType_relationship_name", None)

        if wide_search_limit is not None:
            relevant_ids_to_filter = list(
                {
                    str(getattr(scored_node, "id"))
                    for collection_name, score_collection in node_distances.items()
                    if collection_name != "EdgeType_relationship_name"
                    and isinstance(score_collection, (list, tuple))
                    for scored_node in score_collection
                    if getattr(scored_node, "id", None)
                }
            )
        else:
            relevant_ids_to_filter = None
        
        step2_duration = (time.time() - step2_start) * 1000
        logger.info(
            f"⏱️ [STEP 2] Build node_distances: {step2_duration:.2f}ms ({len(relevant_ids_to_filter or [])} unique IDs)"
        )

        # ========== STEP 3: Get Memory Fragment (Graph Projection) ==========
        step3_start = time.time()
        
        if memory_fragment is None:
            memory_fragment = await get_memory_fragment(
                properties_to_project=properties_to_project,
                node_type=node_type,
                node_name=node_name,
                relevant_ids_to_filter=relevant_ids_to_filter,
                triplet_distance_penalty=triplet_distance_penalty,
            )
        
        step3_duration = (time.time() - step3_start) * 1000
        logger.info(
            f"⏱️ [STEP 3] Graph Projection (get_memory_fragment): {step3_duration:.2f}ms"
        )

        # ========== STEP 4: Map vector distances to nodes ==========
        step4_start = time.time()
        
        await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)
        
        step4_duration = (time.time() - step4_start) * 1000
        logger.info(
            f"⏱️ [STEP 4] Map distances to nodes: {step4_duration:.2f}ms"
        )

        # ========== STEP 5: Map vector distances to edges ==========
        step5_start = time.time()
        
        await memory_fragment.map_vector_distances_to_graph_edges(edge_distances=edge_distances)
        
        step5_duration = (time.time() - step5_start) * 1000
        logger.info(
            f"⏱️ [STEP 5] Map distances to edges: {step5_duration:.2f}ms"
        )

        # ========== STEP 6: Calculate top triplet importances ==========
        step6_start = time.time()
        
        results = await memory_fragment.calculate_top_triplet_importances(k=top_k)
        
        step6_duration = (time.time() - step6_start) * 1000
        logger.info(
            f"⏱️ [STEP 6] Calculate triplet importances: {step6_duration:.2f}ms (top_k={top_k})"
        )

        # ========== SUMMARY ==========
        total_duration = step1_duration + step2_duration + step3_duration + step4_duration + step5_duration + step6_duration
        logger.info(
            f"⏱️ [TOTAL] brute_force_triplet_search: {total_duration:.2f}ms"
        )

        return results

    except CollectionNotFoundError:
        return []
    except Exception as error:
        logger.error(
            "Error during brute force search for query: %s. Error: %s",
            query,
            error,
        )
        raise error

