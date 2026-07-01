"""LLM Generator and Citation Verifier for the RAG pipeline.

Supports NVIDIA NIM (Llama), OpenAI-compatible endpoints, and a local
fallback mode that summarises retrieved chunks without an API key.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Core LLM client helper
# ---------------------------------------------------------------------------

def _build_client():
    """Return (client, model) or (None, None) if no API key is configured."""
    try:
        from openai import OpenAI

        nvidia_key = os.getenv("NVIDIA_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if nvidia_key:
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
            )
            return client, os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")

        if openai_key:
            client = OpenAI(api_key=openai_key)
            return client, os.getenv("LLM_MODEL", "gpt-4o-mini")

    except ImportError:
        pass

    return None, None


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Generate grounded answers with inline citations from retrieved chunks."""

    def __init__(self):
        self.client, self.model = _build_client()
        if self.client:
            print(f"✓ LLM generator ready ({self.model})")
        else:
            print("⚠ No API key found — using local fallback generator")

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer from retrieved chunks.

        Returns a dict with:
          - answer      : str
          - citations   : List[int]  (1-based chunk indices referenced)
          - model       : str
          - chunk_count : int
        """
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "citations": [],
                "model": "none",
                "chunk_count": 0,
            }

        # Build numbered context blocks
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", chunk.get("content", ""))
            src  = chunk.get("metadata", {}).get("source", "unknown")
            context_blocks.append(f"[{i}] (source: {src})\n{text}")

        context_str = "\n\n".join(context_blocks)

        if self.client:
            return self._llm_generate(query, context_str, len(chunks))
        else:
            return self._fallback_generate(query, chunks, context_str)

    # ------------------------------------------------------------------
    def _llm_generate(self, query: str, context_str: str, chunk_count: int) -> Dict:
        system_prompt = (
            "You are a precise RAG assistant. Answer the question using ONLY "
            "the numbered context blocks provided. Cite each block you use with "
            "its number in square brackets, e.g. [1], [2]. "
            "If the context does not contain the answer, say exactly: "
            "'I don't have enough information in the provided documents.'"
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            answer = resp.choices[0].message.content.strip()
            citations = _parse_citations(answer)
            return {
                "answer": answer,
                "citations": citations,
                "model": self.model,
                "chunk_count": chunk_count,
            }
        except Exception as e:
            return {
                "answer": f"LLM error: {e}",
                "citations": [],
                "model": self.model,
                "chunk_count": chunk_count,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    def _fallback_generate(self, query: str, chunks: List[Dict], context_str: str) -> Dict:
        """No-API fallback: return the most relevant chunk text as the answer."""
        best = chunks[0] if chunks else {}
        text = best.get("text", best.get("content", "No content found."))
        answer = f"[1] {text[:500]}"
        return {
            "answer": answer,
            "citations": [1],
            "model": "local-fallback",
            "chunk_count": len(chunks),
        }


# ---------------------------------------------------------------------------
# CitationVerifier
# ---------------------------------------------------------------------------

class CitationVerifier:
    """
    Verify that each cited chunk actually supports the claim in the answer.

    Uses word-overlap heuristic when no LLM is available; LLM-as-judge
    when an API key is present.
    """

    def __init__(self):
        self.client, self.model = _build_client()

    def verify(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns:
          - coverage      : float  (fraction of citations that are supported)
          - all_verified  : bool
          - unsupported   : List[int]  (1-based citation indices that failed)
          - details       : List[Dict]
        """
        citations = _parse_citations(answer)
        if not citations:
            return {
                "coverage": 1.0,
                "all_verified": True,
                "unsupported": [],
                "details": [],
            }

        details = []
        unsupported = []

        for cite_idx in citations:
            chunk_idx = cite_idx - 1  # convert 1-based → 0-based
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                unsupported.append(cite_idx)
                details.append({"citation": cite_idx, "supported": False, "reason": "out of range"})
                continue

            chunk_text = chunks[chunk_idx].get("text", chunks[chunk_idx].get("content", ""))

            if self.client:
                supported, reason = self._llm_verify(answer, chunk_text, cite_idx)
            else:
                supported, reason = self._overlap_verify(answer, chunk_text)

            details.append({"citation": cite_idx, "supported": supported, "reason": reason})
            if not supported:
                unsupported.append(cite_idx)

        coverage = (len(citations) - len(unsupported)) / len(citations) if citations else 1.0

        return {
            "coverage": coverage,
            "all_verified": len(unsupported) == 0,
            "unsupported": unsupported,
            "details": details,
        }

    # ------------------------------------------------------------------
    def _overlap_verify(self, answer: str, chunk_text: str) -> tuple:
        answer_words  = set(re.sub(r"[^\w\s]", "", answer.lower()).split())
        chunk_words   = set(re.sub(r"[^\w\s]", "", chunk_text.lower()).split())
        # Remove very common stop words from consideration
        stop = {"the", "a", "an", "is", "are", "was", "in", "of", "to", "and", "or", "it"}
        answer_words -= stop
        chunk_words  -= stop

        if not answer_words:
            return True, "empty answer"

        overlap = len(answer_words & chunk_words) / len(answer_words)
        supported = overlap > 0.10  # 10% threshold — generous for short chunks
        return supported, f"word overlap {overlap:.1%}"

    # ------------------------------------------------------------------
    def _llm_verify(self, answer: str, chunk_text: str, cite_idx: int) -> tuple:
        prompt = (
            f"Does the following source chunk support the claim made in the answer?\n\n"
            f"Source chunk:\n{chunk_text}\n\n"
            f"Answer excerpt (citing [{cite_idx}]):\n{answer}\n\n"
            "Reply with exactly YES or NO, then a one-sentence reason."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            reply = resp.choices[0].message.content.strip().upper()
            has_yes = "YES" in reply
            has_no  = "NO"  in reply
            if has_yes and not has_no:
                return True,  reply
            if has_no and not has_yes:
                return False, reply
            # Ambiguous — fall back to word overlap heuristic
            return self._overlap_verify(answer, chunk_text)
        except Exception as e:
            return self._overlap_verify(answer, chunk_text)


# ---------------------------------------------------------------------------
# Legacy LLMGenerator kept for backward compatibility
# ---------------------------------------------------------------------------

class LLMGenerator:
    """Legacy wrapper — prefer AnswerGenerator for new code."""

    def __init__(self, model: Optional[str] = None):
        self._gen = AnswerGenerator()

    def generate(self, context: str, question: str) -> str:
        # Wrap plain string context into a fake chunk list
        chunks = [{"text": context, "metadata": {"source": "context"}}]
        result = self._gen.generate(question, chunks)
        return result["answer"]


# ---------------------------------------------------------------------------
# Module-level convenience functions (used by API layer)
# ---------------------------------------------------------------------------

_generator: Optional[AnswerGenerator] = None
_verifier:  Optional[CitationVerifier] = None


def generate_answer(query: str, chunks: List[Dict]) -> Dict:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator.generate(query, chunks)


def verify_citations(answer: str, chunks: List[Dict]) -> Dict:
    global _verifier
    if _verifier is None:
        _verifier = CitationVerifier()
    return _verifier.verify(answer, chunks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_citations(text: str) -> List[int]:
    """Extract all [N] citation numbers from a string."""
    return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))