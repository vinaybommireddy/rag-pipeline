from typing import List
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    chunk_id: str
    embedding: List[float]
    model: str

class EmbeddingsManager:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = model
        print(f"Loading: {model}")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model)
    
    def embed_text(self, text: str) -> List[float]:
        embedding = self.embedder.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List) -> List[EmbeddingResult]:
        results = []
        for i, chunk in enumerate(chunks):
            try:
                emb = self.embed_text(chunk.content)
                results.append(EmbeddingResult(f"{chunk.source}_{chunk.chunk_id}", emb, self.model))
                if (i+1) % 5 == 0:
                    print(f"✓ Embedded {i+1}/{len(chunks)}")
            except Exception as e:
                print(f"✗ Error: {e}")
        print(f"✓ Total: {len(results)} chunks")
        return results