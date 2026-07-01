"""Hybrid retrieval engine combining dense (vector) + sparse (BM25) search."""
from typing import List, Dict, Any, Optional
from collections import defaultdict


class HybridRetriever:
    """End-to-end hybrid retrieval with dense, sparse, and RRF fusion."""

    def __init__(self, vector_store=None, bm25=None, embedder=None):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Full hybrid retrieval: dense + sparse + RRF fusion."""

        try:
            # 1. Dense retrieval (vector search)
            query_emb = self.embedder.embed_text(query)
            dense_results_raw = self.vector_store.query(query_embedding=query_emb, n_results=top_k)

            dense_results = []
            if isinstance(dense_results_raw, dict):
                # ChromaDB returns nested lists: ids[0], documents[0], etc.
                ids       = dense_results_raw.get('ids', [[]])[0]
                documents = dense_results_raw.get('documents', [[]])[0]
                metadatas = dense_results_raw.get('metadatas', [[]])[0]
                distances = dense_results_raw.get('distances', [[]])[0]
                for i, chunk_id in enumerate(ids):
                    dense_results.append({
                        "text":     documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "score":    1 - distances[i] if i < len(distances) else 0,
                        "rank":     i + 1,
                        "source":   "dense"
                    })

            # 2. Sparse retrieval (BM25)
            sparse_results = []
            if self.bm25:
                raw_sparse = self.bm25.search(query, top_k=top_k)
                for i, r in enumerate(raw_sparse):
                    sparse_results.append({
                        "text": r.get("content", ""),
                        "metadata": r.get("metadata", {}),
                        "score": r.get("score", 0),
                        "rank": i + 1,
                        "source": "sparse"
                    })

            # 3. RRF Fusion
            fused = self._rrf_fuse(dense_results, sparse_results, top_k=top_k * 2)

            return {
                "query": query,
                "dense_results": dense_results,
                "sparse_results": sparse_results,
                "fused_results": fused,
                "top_results": fused[:top_k]
            }

        except Exception as e:
            print(f"✗ Retrieval error: {e}")
            return {
                "query": query,
                "dense_results": [],
                "sparse_results": [],
                "fused_results": [],
                "top_results": [],
                "error": str(e)
            }

    def _rrf_fuse(self, dense: List[Dict], sparse: List[Dict], top_k: int = 20, k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion — combines dense and sparse ranked lists."""

        scores = defaultdict(float)
        chunk_data = {}

        # Dense weight = 0.7, Sparse weight = 0.3
        for r in dense:
            key = r["text"][:100]  # Use truncated text as key
            scores[key] += 0.7 / (k + r.get("rank", 1))
            if key not in chunk_data:
                chunk_data[key] = r

        for r in sparse:
            key = r["text"][:100]
            scores[key] += 0.3 / (k + r.get("rank", 1))
            if key not in chunk_data:
                chunk_data[key] = r

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "text": chunk_data[text].get("text", ""),
                "metadata": chunk_data[text].get("metadata", {}),
                "rrf_score": score,
                "rank": i + 1,
                "source": chunk_data[text].get("source", "fused")
            }
            for i, (text, score) in enumerate(sorted_chunks[:top_k])
        ]


# ---------------------------------------------------------------------------
# Global retriever instance (used by API layer)
# ---------------------------------------------------------------------------
_retriever = None


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """Standalone retrieve function for the API."""
    global _retriever

    if _retriever is None:
        from src.vector_store import VectorStore
        from src.bm25_search import BM25Search
        from src.embeddings_manager import EmbeddingsManager

        print("Initializing retriever...")
        vs = VectorStore()
        bm25 = BM25Search()
        bm25.load_index()
        em = EmbeddingsManager()

        _retriever = HybridRetriever(vs, bm25, em)
        print("✓ Retriever initialized")

    result = _retriever.retrieve(query, top_k)
    return result.get("top_results", [])