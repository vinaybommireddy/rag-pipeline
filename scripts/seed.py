from typing import List
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class EmbeddingResult:
    chunk_id: str
    embedding: List[float]
    model: str

class EmbeddingsManager:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = model
        print(f"✓ Loading sentence-transformers model: {model}")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model)
            print(f"✓ Model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> List[float]:
        embedding = self.embedder.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List) -> List[EmbeddingResult]:
        results = []
        for i, chunk in enumerate(chunks):
            try:
                emb = self.embed_text(chunk.content)
                results.append(EmbeddingResult(
                    chunk_id=f"{chunk.source}_{chunk.chunk_id}",
                    embedding=emb,
                    model=self.model
                ))
                if (i+1) % 5 == 0:
                    print(f"✓ Embedded {i+1}/{len(chunks)}")
            except Exception as e:
                print(f"✗ Error embedding chunk {i}: {e}")
        print(f"✓ Successfully embedded {len(results)}/{len(chunks)} chunks")
        return results