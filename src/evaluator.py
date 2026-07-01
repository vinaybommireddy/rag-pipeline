"""
Phase 4: Evaluation Framework

Automated eval metrics for the RAG pipeline:
  1. Answer Correctness   — LLM-as-judge vs golden answer (or keyword overlap fallback)
  2. Faithfulness         — are all answer claims grounded in retrieved context?
  3. Retrieval Relevance  — did retrieval return the right source documents?
  4. Citation Accuracy    — do citations actually support claims?

Run: python -m src.evaluator
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Metric scorers
# ---------------------------------------------------------------------------

class AnswerCorrectnessScorer:
    """Score how correct the generated answer is vs the golden answer."""

    def __init__(self, client=None, model: str = ""):
        self.client = client
        self.model  = model

    def score(self, question: str, generated: str, golden: str) -> Dict:
        if golden == "NOT_IN_CORPUS":
            # For unanswerable questions: correct if system abstains or says IDK
            idk_signals = ["don't have", "not enough", "cannot", "no information",
                           "not found", "unable", "abstain", "i don't know"]
            abstained = any(s in generated.lower() for s in idk_signals)
            return {
                "score":      1.0 if abstained else 0.0,
                "method":     "unanswerable-check",
                "correct":    abstained,
                "explanation": "Correctly abstained" if abstained else "Should have said IDK"
            }

        if self.client:
            return self._llm_score(question, generated, golden)
        return self._overlap_score(generated, golden)

    def _llm_score(self, question: str, generated: str, golden: str) -> Dict:
        prompt = (
            f"Question: {question}\n"
            f"Golden answer: {golden}\n"
            f"Generated answer: {generated}\n\n"
            "Score the generated answer vs the golden answer from 0.0 to 1.0.\n"
            "1.0 = fully correct and complete\n"
            "0.5 = partially correct\n"
            "0.0 = wrong or missing key info\n"
            "Reply with ONLY a number like: 0.8"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            score = float(re.search(r"\d+\.?\d*", text).group())
            score = max(0.0, min(1.0, score))
            return {"score": score, "method": "llm-judge", "correct": score >= 0.6, "explanation": text}
        except Exception as e:
            return self._overlap_score(generated, golden)

    def _overlap_score(self, generated: str, golden: str) -> Dict:
        stop = {"the","a","an","is","are","was","in","of","to","and","or","it","for"}
        g_words = set(re.sub(r"[^\w\s]", "", golden.lower()).split()) - stop
        a_words = set(re.sub(r"[^\w\s]", "", generated.lower()).split()) - stop
        if not g_words:
            return {"score": 1.0, "method": "overlap", "correct": True, "explanation": "empty golden"}
        overlap = len(g_words & a_words) / len(g_words)
        return {
            "score":       round(overlap, 3),
            "method":      "keyword-overlap",
            "correct":     overlap >= 0.4,
            "explanation": f"{overlap:.0%} keyword overlap"
        }


class FaithfulnessScorer:
    """Score whether the answer is grounded in the retrieved chunks."""

    def score(self, answer: str, chunks: List[Dict]) -> Dict:
        if not chunks or not answer:
            return {"score": 0.0, "grounded": False, "explanation": "no chunks or answer"}

        # Combine all chunk text
        context = " ".join(c.get("text", c.get("content", "")) for c in chunks).lower()
        stop = {"the","a","an","is","are","was","in","of","to","and","or","it","for","with"}

        # Split answer into sentences and check each
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 15]
        if not sentences:
            return {"score": 1.0, "grounded": True, "explanation": "no claims to check"}

        grounded_count = 0
        details = []
        for sent in sentences:
            words = set(re.sub(r"[^\w\s]", "", sent.lower()).split()) - stop
            if not words:
                continue
            found = len(words & set(context.split())) / len(words)
            grounded = found > 0.25
            if grounded:
                grounded_count += 1
            details.append({"sentence": sent[:60] + "...", "overlap": round(found, 3), "grounded": grounded})

        score = grounded_count / len(sentences) if sentences else 1.0
        return {
            "score":       round(score, 3),
            "grounded":    score >= 0.6,
            "explanation": f"{grounded_count}/{len(sentences)} sentences grounded",
            "details":     details
        }


class RetrievalRelevanceScorer:
    """Score whether the right source documents were retrieved."""

    def score(self, chunks: List[Dict], relevant_sources: List[str]) -> Dict:
        if not relevant_sources:
            # Unanswerable: ideally retriever finds nothing confident
            return {"score": 1.0, "hit_rate": 1.0, "explanation": "unanswerable — no sources expected"}

        if not chunks:
            return {"score": 0.0, "hit_rate": 0.0, "explanation": "no chunks retrieved"}

        retrieved_sources = set()
        for c in chunks:
            src = c.get("metadata", {}).get("source", "")
            # Normalize — strip path separators
            src_base = Path(src).name if src else ""
            retrieved_sources.add(src_base)

        expected = set(Path(s).name for s in relevant_sources)
        hits = len(expected & retrieved_sources)
        hit_rate = hits / len(expected) if expected else 1.0

        return {
            "score":            round(hit_rate, 3),
            "hit_rate":         round(hit_rate, 3),
            "expected_sources": list(expected),
            "retrieved_sources": list(retrieved_sources),
            "hits":             hits,
            "explanation":      f"{hits}/{len(expected)} expected sources retrieved"
        }


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Runs the full eval suite over golden_qa.json and produces a report.
    """

    def __init__(self, golden_path: str = "data/eval/golden_qa.json"):
        self.golden_path = golden_path
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Load all RAG components."""
        from src.vector_store       import VectorStore
        from src.bm25_search        import BM25Search
        from src.embeddings_manager import EmbeddingsManager
        from src.retriever          import HybridRetriever
        from src.generator          import AnswerGenerator, CitationVerifier, _build_client

        print("Loading RAG pipeline for evaluation...")
        vs   = VectorStore()
        bm25 = BM25Search()
        bm25.load_index()
        em   = EmbeddingsManager()

        self.retriever = HybridRetriever(vs, bm25, em)
        self.generator = AnswerGenerator()
        self.verifier  = CitationVerifier()

        # Metric scorers
        client, model = _build_client()
        self.correctness_scorer = AnswerCorrectnessScorer(client, model)
        self.faithfulness_scorer = FaithfulnessScorer()
        self.relevance_scorer    = RetrievalRelevanceScorer()
        print("✓ Pipeline loaded\n")

    def run(self, strategy: str = "fixed-size", top_k: int = 3) -> Dict:
        """Run the full evaluation suite."""
        with open(self.golden_path, "r") as f:
            questions = json.load(f)

        print(f"Running eval on {len(questions)} questions (strategy={strategy}) ...")
        print("-" * 60)

        results = []
        for i, qa in enumerate(questions):
            result = self._eval_one(qa, top_k=top_k)
            results.append(result)

            # Progress indicator
            cat = qa.get("category", "?")[:10]
            diff = qa.get("difficulty", "?")[:12]
            corr = result["correctness"]["score"]
            faith = result["faithfulness"]["score"]
            print(f"  [{i+1:02d}/{len(questions)}] {qa['id']} [{diff:<12}] "
                  f"correctness={corr:.0%}  faithfulness={faith:.0%}")

        report = self._build_report(results, strategy)
        return report

    def _eval_one(self, qa: Dict, top_k: int = 3) -> Dict:
        """Evaluate a single Q&A pair."""
        question = qa["question"]
        golden   = qa["golden_answer"]

        # 1. Retrieve
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        chunks    = retrieval.get("top_results", [])

        # 2. Generate
        gen_out = self.generator.generate(question, chunks)
        answer  = gen_out.get("answer", "")

        # 3. Verify citations
        verif = self.verifier.verify(answer, chunks)

        # 4. Score
        correctness  = self.correctness_scorer.score(question, answer, golden)
        faithfulness = self.faithfulness_scorer.score(answer, chunks)
        relevance    = self.relevance_scorer.score(chunks, qa.get("relevant_sources", []))

        return {
            "id":            qa["id"],
            "category":      qa.get("category", "unknown"),
            "difficulty":    qa.get("difficulty", "unknown"),
            "question":      question,
            "golden_answer": golden,
            "generated_answer": answer,
            "citations":     gen_out.get("citations", []),
            "correctness":   correctness,
            "faithfulness":  faithfulness,
            "retrieval_relevance": relevance,
            "citation_accuracy":   verif,
        }

    def _build_report(self, results: List[Dict], strategy: str) -> Dict:
        """Aggregate results into a summary report."""

        def avg(key_path):
            vals = []
            for r in results:
                parts = key_path.split(".")
                v = r
                for p in parts:
                    v = v.get(p, {}) if isinstance(v, dict) else None
                if isinstance(v, (int, float)):
                    vals.append(v)
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        # By category
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        cat_summary = {}
        for cat, items in categories.items():
            cat_summary[cat] = {
                "count":        len(items),
                "correctness":  round(sum(i["correctness"]["score"] for i in items) / len(items), 3),
                "faithfulness": round(sum(i["faithfulness"]["score"] for i in items) / len(items), 3),
            }

        # By difficulty
        difficulties = {}
        for r in results:
            diff = r["difficulty"]
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(r)

        diff_summary = {}
        for diff, items in difficulties.items():
            diff_summary[diff] = {
                "count":       len(items),
                "correctness": round(sum(i["correctness"]["score"] for i in items) / len(items), 3),
            }

        report = {
            "strategy":    strategy,
            "timestamp":   datetime.now().isoformat(),
            "total_questions": len(results),
            "overall": {
                "answer_correctness":    avg("correctness.score"),
                "faithfulness":          avg("faithfulness.score"),
                "retrieval_relevance":   avg("retrieval_relevance.score"),
                "citation_accuracy":     avg("citation_accuracy.coverage"),
            },
            "by_category":   cat_summary,
            "by_difficulty": diff_summary,
            "results":       results,
        }
        return report


# ---------------------------------------------------------------------------
# Chunking Strategy Comparison
# ---------------------------------------------------------------------------

class ChunkingStrategyComparison:
    """
    Run the eval suite across multiple chunking strategies and compare.
    Generates a side-by-side comparison report.
    """

    STRATEGIES = ["fixed-size", "recursive"]   # "semantic" requires more tokens

    def run(self) -> Dict:
        comparison = {}
        for strategy in self.STRATEGIES:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            print(f"{'='*60}")
            # Re-ingest with this strategy
            self._reingest(strategy)
            evaluator = RAGEvaluator()
            report    = evaluator.run(strategy=strategy)
            comparison[strategy] = report["overall"]
            comparison[strategy]["by_category"] = report["by_category"]

        return comparison

    def _reingest(self, strategy: str):
        """Re-chunk and re-index with a different chunking strategy."""
        from src.ingest             import DocumentLoader
        from src.chunking_strategies import ChunkingFactory
        from src.embeddings_manager import EmbeddingsManager
        from src.vector_store       import VectorStore
        from src.bm25_search        import BM25Search

        loader = DocumentLoader()
        docs   = loader.load_from_directory()
        chunks = []
        for doc in docs:
            chunks.extend(ChunkingFactory.create_chunks(doc.content, doc.source, strategy=strategy))

        em  = EmbeddingsManager()
        vs  = VectorStore()
        vs.reset()  # clear existing

        embedded = em.embed_chunks(chunks)
        embeddings = [r.embedding for r in embedded]
        if embeddings:
            vs.add_chunks(chunks[:len(embedded)], embeddings)

        bm25 = BM25Search()
        bm25.build_index(chunks[:len(embedded)])
        print(f"  ✓ Re-indexed {len(chunks)} chunks with strategy={strategy}")


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_report(report: Dict):
    """Pretty-print an evaluation report to the console."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT — Strategy: {report['strategy']}")
    print(f"Questions: {report['total_questions']}  |  {report['timestamp'][:19]}")
    print(f"{'='*60}")

    ov = report["overall"]
    print(f"\n📊 OVERALL METRICS")
    print(f"  Answer Correctness  : {ov['answer_correctness']:.1%}")
    print(f"  Faithfulness        : {ov['faithfulness']:.1%}")
    print(f"  Retrieval Relevance : {ov['retrieval_relevance']:.1%}")
    print(f"  Citation Accuracy   : {ov['citation_accuracy']:.1%}")

    print(f"\n📂 BY CATEGORY")
    for cat, data in report.get("by_category", {}).items():
        print(f"  {cat:<15} n={data['count']}  "
              f"correctness={data['correctness']:.0%}  "
              f"faithfulness={data['faithfulness']:.0%}")

    print(f"\n🎯 BY DIFFICULTY")
    for diff, data in report.get("by_difficulty", {}).items():
        print(f"  {diff:<14} n={data['count']}  correctness={data['correctness']:.0%}")


def save_report(report: Dict, path: str = "data/eval/report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Full report saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_eval(strategy: str = "fixed-size"):
    """Run evaluation and print + save the report."""
    evaluator = RAGEvaluator()
    report    = evaluator.run(strategy=strategy)
    print_report(report)
    save_report(report)
    return report


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed-size"
    run_eval(strategy)
