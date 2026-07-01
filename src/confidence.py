"""
Phase 3: Answer Confidence Scorer

Scores each RAG answer on three dimensions:
  1. retrieval_confidence  — how relevant were the top chunks?
  2. citation_coverage     — what % of citations are verified?
  3. answer_completeness   — did the answer address all question parts?

Returns a composite score 0.0 → 1.0 and an "I don't know" decision.
"""

from typing import List, Dict, Any
import re


# Threshold below which we refuse to answer and return the IDK response
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.25
CITATION_COVERAGE_THRESHOLD    = 0.30


class AnswerConfidenceScorer:
    """Scores a RAG answer across three dimensions and decides if it's trustworthy."""

    def score(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        query        : the original user question
        answer       : the generated answer text
        chunks       : top retrieved chunks (each has 'text', 'rrf_score' or 'score')
        verification : output of CitationVerifier.verify()

        Returns
        -------
        {
          "retrieval_confidence": float,   # avg similarity of top chunks
          "citation_coverage":    float,   # % verified citations
          "answer_completeness":  float,   # % question words found in answer
          "composite_score":      float,   # weighted average
          "trustworthy":          bool,    # above threshold?
          "reason":               str,     # human-readable verdict
        }
        """
        rc = self._retrieval_confidence(chunks)
        cc = verification.get("coverage", 0.0)
        ac = self._answer_completeness(query, answer)

        # Weighted composite: retrieval matters most
        composite = 0.50 * rc + 0.30 * cc + 0.20 * ac

        trustworthy = rc >= RETRIEVAL_CONFIDENCE_THRESHOLD
        reason = self._build_reason(rc, cc, ac, trustworthy)

        return {
            "retrieval_confidence": round(rc, 3),
            "citation_coverage":    round(cc, 3),
            "answer_completeness":  round(ac, 3),
            "composite_score":      round(composite, 3),
            "trustworthy":          trustworthy,
            "reason":               reason,
        }

    # ------------------------------------------------------------------
    def _retrieval_confidence(self, chunks: List[Dict]) -> float:
        """Average of the top-3 chunk scores (rrf_score or cosine score)."""
        if not chunks:
            return 0.0
        top = chunks[:3]
        scores = []
        for c in top:
            s = c.get("rrf_score") or c.get("score") or 0.0
            scores.append(float(s))
        if not scores:
            return 0.0
        # Normalise: rrf scores are tiny (e.g. 0.011), cosine scores are 0-1
        avg = sum(scores) / len(scores)
        # If all scores look like rrf (< 0.1), scale up to 0-1 range
        if avg < 0.1:
            avg = min(avg * 20, 1.0)
        return min(avg, 1.0)

    def _answer_completeness(self, query: str, answer: str) -> float:
        """Fraction of meaningful query words that appear in the answer."""
        stop = {"what","is","are","the","a","an","how","does","do","in","of",
                "to","and","or","it","was","were","for","with","can","you"}
        q_words = set(re.sub(r"[^\w\s]", "", query.lower()).split()) - stop
        if not q_words:
            return 1.0
        a_words = set(re.sub(r"[^\w\s]", "", answer.lower()).split())
        hit = len(q_words & a_words) / len(q_words)
        return min(hit, 1.0)

    def _build_reason(self, rc: float, cc: float, ac: float, trustworthy: bool) -> str:
        if not trustworthy:
            return (
                f"Low retrieval confidence ({rc:.0%}) — relevant documents may not be "
                "in the corpus. Consider checking sources manually."
            )
        parts = []
        if cc < 0.5:
            parts.append(f"citation coverage is low ({cc:.0%})")
        if ac < 0.5:
            parts.append(f"answer may not address all question parts ({ac:.0%} completeness)")
        if parts:
            return "Answer generated but: " + "; ".join(parts) + "."
        return f"Answer is well-supported (retrieval {rc:.0%}, citations {cc:.0%})."


# ---------------------------------------------------------------------------
# "I Don't Know" Handler
# ---------------------------------------------------------------------------

class IDontKnowHandler:
    """
    Returns a structured 'I don't know' response when confidence is too low.

    This is more useful than a hallucinated answer and is a strong
    interview signal for production RAG maturity.
    """

    def should_abstain(self, confidence: Dict[str, Any]) -> bool:
        return not confidence.get("trustworthy", True)

    def build_response(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        confidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a structured abstention response."""

        found_topics = []
        for c in chunks[:3]:
            src = c.get("metadata", {}).get("source", "unknown")
            snippet = c.get("text", "")[:80].strip()
            if snippet:
                found_topics.append({"source": src, "preview": snippet + "..."})

        docs_to_check = sorted(
            set(c.get("metadata", {}).get("source", "") for c in chunks if c)
        )

        return {
            "answer": (
                f"I don't have enough information in the indexed documents to "
                f"confidently answer: \"{query}\""
            ),
            "citations": [],
            "confidence": confidence,
            "what_was_found": found_topics,
            "documents_to_check_manually": docs_to_check,
            "suggestion": (
                "Try rephrasing your question, or add more relevant documents "
                "to the data/documents/ folder and re-index."
            ),
            "abstained": True,
        }
