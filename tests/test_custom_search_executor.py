from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.search.custom.executor import SearchExecutor


def test_graph_executor_uses_nodeset_type_by_default():
    executor = SearchExecutor(initial_top_k=10)
    assert executor.graph_retriever.node_type is NodeSet
