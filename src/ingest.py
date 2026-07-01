"""Document ingestion module for RAG pipeline"""

from pathlib import Path
from typing import List, Dict
import os

class Document:
    """Simple document class"""
    def __init__(self, content: str, source: str, metadata: Dict = None):
        self.content = content
        self.source = source
        self.metadata = metadata or {}

class DocumentLoader:
    """Load documents from various formats"""
    
    def __init__(self, data_dir: str = "./data/documents"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_text_file(self, file_path: str) -> Document:
        """Load .txt file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            source=str(file_path),
            metadata={"format": "text"}
        )
    
    def load_markdown_file(self, file_path: str) -> Document:
        """Load .md file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            source=str(file_path),
            metadata={"format": "markdown"}
        )
    
    def load_from_directory(self) -> List[Document]:
        """Load all .txt and .md files from data directory"""
        documents = []
        
        for file_path in self.data_dir.glob("**/*.txt"):
            try:
                doc = self.load_text_file(str(file_path))
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        for file_path in self.data_dir.glob("**/*.md"):
            try:
                doc = self.load_markdown_file(str(file_path))
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

def ingest(file_content: str, filename: str) -> Dict:
    """Ingest a document into the RAG system"""
    
    try:
        doc = Document(content=file_content, source=filename)
        
        from src.chunking_strategies import ChunkingFactory
        from src.embeddings_manager import EmbeddingsManager
        from src.vector_store import VectorStore
        from src.bm25_search import BM25Search
        
        # Chunk
        chunks = ChunkingFactory.create_chunks(doc.content, doc.source, strategy="fixed-size")
        
        # Embed
        em = EmbeddingsManager()
        results = em.embed_chunks(chunks)
        embeddings = [r.embedding for r in results]
        
        # Store
        vs = VectorStore()
        vs.add_chunks(chunks[:len(results)], embeddings)
        
        # Index
        bm25 = BM25Search()
        bm25.build_index(chunks[:len(results)])
        
        return {
            "status": "success",
            "filename": filename,
            "chunks": len(chunks),
            "embeddings": len(results)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }