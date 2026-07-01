"""ChromaDB vector store for dense retrieval."""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB wrapper for storing and querying embeddings."""
    
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "rag_docs",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Any], embeddings: List[List[float]]):
        """Add chunks with embeddings to the store."""
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'source'):
                source = chunk.source
            elif isinstance(chunk, dict):
                source = chunk.get('source', 'unknown')
            else:
                source = 'unknown'
            
            if hasattr(chunk, 'chunk_index'):
                idx = chunk.chunk_index
            elif isinstance(chunk, dict):
                idx = chunk.get('chunk_index', i)
            else:
                idx = i
            
            if hasattr(chunk, 'text'):
                text = chunk.text
            elif hasattr(chunk, 'content'):
                text = chunk.content
            elif isinstance(chunk, dict):
                text = chunk.get('text', chunk.get('content', str(chunk)))
            else:
                text = str(chunk)
            
            chunk_id = f"{source}_{idx}"
            ids.append(chunk_id)
            texts.append(text)
            metadatas.append({
                "source": source,
                "chunk_index": idx,
            })
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    def query(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """Query the vector store with an embedding."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all stored chunks."""
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
    
    def count(self) -> int:
        return self.collection.count()
    
    def reset(self):
        """Clear all data."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
