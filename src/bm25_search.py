from rank_bm25 import BM25Okapi
from typing import List, Dict
import pickle
import os

class BM25Search:
    def __init__(self, index_path: str = "./data/bm25_index.pkl"):
        self.index_path = index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    def build_index(self, chunks: List):
        self.documents = chunks
        self.doc_ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]

        corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)

        self._save_index()
        print(f"✓ Built BM25 index with {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        retrieved = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk = self.documents[idx]
                retrieved.append({
                    'rank': rank + 1,
                    'chunk_id': self.doc_ids[idx],
                    'content': chunk.content,
                    'score': float(scores[idx]),
                    'source': chunk.source,
                    'metadata': chunk.metadata
                })

        return retrieved

    def _save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'doc_ids': self.doc_ids,
                'chunks': self.documents  # ✅ save full chunks so retriever has text
            }, f)
        print(f"✓ Saved BM25 index to {self.index_path}")

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.doc_ids = data['doc_ids']
                self.documents = data.get('chunks', [])
            print(f"✓ Loaded BM25 index from {self.index_path}")
        else:
            print(f"✗ Index not found at {self.index_path}")