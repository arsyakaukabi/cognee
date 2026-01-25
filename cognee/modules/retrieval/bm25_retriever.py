"""
BM25 Retriever implementation using rank-bm25 library.

This module provides BM25-based lexical retrieval that extends the LexicalRetriever
base class. It builds an in-memory BM25 index from DocumentChunks and supports
configurable tokenization.
"""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from cognee.modules.retrieval.lexical_retriever import LexicalRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger("BM25Retriever")


class BM25Retriever(LexicalRetriever):
    """
    BM25-based lexical retriever using the rank-bm25 library.

    This retriever builds an in-memory BM25 index from DocumentChunks and provides
    efficient lexical search capabilities. It's designed to complement vector-based
    search for hybrid retrieval scenarios.

    Attributes:
        top_k: Maximum number of results to return.
        with_scores: If True, return (payload, score) tuples.
        bm25_index: The BM25Okapi index instance.
    """

    def __init__(
        self,
        top_k: int = 10,
        with_scores: bool = False,
        stop_words: Optional[List[str]] = None,
    ):
        """
        Initialize the BM25 retriever.

        Args:
            top_k: Number of top results to return.
            with_scores: If True, return (payload, score) pairs.
            stop_words: Optional list of stop words to filter out during tokenization.
        """
        self.stop_words = {w.lower() for w in stop_words} if stop_words else set()
        self.bm25_index: Optional[BM25Okapi] = None
        self._corpus_ids: List[str] = []

        # Initialize parent with our tokenizer but no scorer
        # (BM25 uses batch scoring, not pairwise)
        super().__init__(
            tokenizer=self._bm25_tokenizer,
            scorer=self._dummy_scorer,
            top_k=top_k,
            with_scores=with_scores,
        )

    def _bm25_tokenizer(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing and querying.

        Performs lowercase conversion, extracts word tokens using regex,
        and filters out stop words.

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase tokens with stop words removed.
        """
        tokens = re.findall(r"\w+", text.lower())
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def _dummy_scorer(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        """
        Placeholder scorer - not used since BM25 uses batch scoring.

        The LexicalRetriever base class requires a scorer, but BM25 computes
        scores for all documents at once via get_scores().
        """
        return 0.0

    async def initialize(self):
        """
        Initialize the BM25 index from DocumentChunks.

        Loads chunks using parent class initialization, then builds the
        BM25Okapi index from the tokenized corpus.
        """
        # Let parent load chunks from graph engine
        await super().initialize()

        if not self.chunks:
            logger.warning("No chunks available to build BM25 index")
            return

        # Build corpus and track IDs
        self._corpus_ids = list(self.chunks.keys())
        corpus = [self.chunks[cid] for cid in self._corpus_ids]

        logger.info("Building BM25 index with %d documents", len(corpus))
        self.bm25_index = BM25Okapi(corpus)
        logger.info("BM25 index built successfully")

    async def get_context(self, query: str) -> List:
        """
        Retrieve relevant chunks using BM25 scoring.

        Args:
            query: The search query string.

        Returns:
            List of chunk payloads (or (payload, score) tuples if with_scores=True),
            ranked by BM25 relevance score.
        """
        if not self._initialized:
            await self.initialize()

        if not self.bm25_index or not self._corpus_ids:
            logger.warning("BM25 index not available, returning empty results")
            return []

        # Tokenize query
        query_tokens = self._bm25_tokenizer(query)
        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization")
            return []

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)

        # Pair IDs with scores and sort by score descending
        scored_results = list(zip(self._corpus_ids, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top_k results
        top_results = scored_results[: self.top_k]

        logger.info(
            "BM25 retrieved %d/%d chunks for query (len=%d tokens)",
            len(top_results),
            len(self._corpus_ids),
            len(query_tokens),
        )

        if self.with_scores:
            return [(self.payloads[chunk_id], score) for chunk_id, score in top_results]
        else:
            return [self.payloads[chunk_id] for chunk_id, _ in top_results]
