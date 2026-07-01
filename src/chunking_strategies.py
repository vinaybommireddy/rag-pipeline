from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: int
    strategy: str
    start_char: int
    end_char: int
    metadata: dict = None

class ChunkingStrategy:
    def chunk(self, text: str, source: str, strategy_name: str) -> List[Chunk]:
        raise NotImplementedError

class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str, strategy_name: str = "fixed-size") -> List[Chunk]:
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            if len(chunk_text.strip()) > 10:
                chunks.append(Chunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    strategy=strategy_name,
                    start_char=i,
                    end_char=min(i + self.chunk_size, len(text)),
                    metadata={"size": len(chunk_text)}
                ))
                chunk_id += 1
        
        return chunks

class RecursiveChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str, strategy_name: str = "recursive") -> List[Chunk]:
        sections = self._split_by_headers(text)
        chunks = []
        chunk_id = 0
        
        for section_title, section_text in sections:
            for i in range(0, len(section_text), self.chunk_size - self.overlap):
                chunk_text = section_text[i:i + self.chunk_size]
                
                if len(chunk_text.strip()) > 10:
                    full_chunk = f"## {section_title}\n{chunk_text}" if section_title else chunk_text
                    chunks.append(Chunk(
                        content=full_chunk,
                        source=source,
                        chunk_id=chunk_id,
                        strategy=strategy_name,
                        start_char=i,
                        end_char=min(i + self.chunk_size, len(section_text)),
                        metadata={"section": section_title}
                    ))
                    chunk_id += 1
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List:
        lines = text.split('\n')
        sections = []
        current_section = ""
        current_title = "Intro"
        
        for line in lines:
            if line.startswith('#'):
                if current_section:
                    sections.append((current_title, current_section))
                current_title = line.replace('#', '').strip()
                current_section = ""
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append((current_title, current_section))
        
        return sections

class ChunkingFactory:
    @staticmethod
    def create_chunks(text: str, source: str, strategy: str = "fixed-size") -> List[Chunk]:
        if strategy == "fixed-size":
            return FixedSizeChunking().chunk(text, source, strategy)
        elif strategy == "recursive":
            return RecursiveChunking().chunk(text, source, strategy)
        else:
            return FixedSizeChunking().chunk(text, source, strategy)