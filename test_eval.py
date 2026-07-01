"""
Phase 4 Test: Evaluation Framework
Run: python test_eval.py
"""
import sys
import traceback
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")

try:
    print("=" * 60)
    print("PHASE 4: EVALUATION FRAMEWORK")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading golden Q&A dataset...")
    import json
    with open("data/eval/golden_qa.json") as f:
        questions = json.load(f)

    cats = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"  ✓ {len(questions)} questions loaded")
    for cat, count in sorted(cats.items()):
        print(f"     {cat:<15} : {count}")

    # ------------------------------------------------------------------ #
    print("\n[2/4] Running full evaluation (fixed-size chunking)...")
    from src.evaluator import RAGEvaluator, print_report, save_report

    evaluator = RAGEvaluator()
    report    = evaluator.run(strategy="fixed-size", top_k=3)
    print_report(report)
    save_report(report, "data/eval/report_fixed_size.json")

    # ------------------------------------------------------------------ #
    print("\n[3/4] Running chunking strategy comparison...")
    from src.evaluator import ChunkingStrategyComparison
    comp = ChunkingStrategyComparison()
    comparison = comp.run()

    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} ", end="")
    strategies = list(comparison.keys())
    for s in strategies:
        print(f"  {s:<15}", end="")
    print()
    print("  " + "-" * (25 + 17 * len(strategies)))

    metrics = ["answer_correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
    for metric in metrics:
        print(f"  {metric:<25}", end="")
        for s in strategies:
            val = comparison[s].get(metric, 0)
            print(f"  {val:.1%}          ", end="")
        print()

    # Determine winner
    winners = {}
    for metric in metrics:
        best_strat = max(strategies, key=lambda s: comparison[s].get(metric, 0))
        best_val   = comparison[best_strat].get(metric, 0)
        winners[metric] = (best_strat, best_val)

    print("\n🏆 WINNERS BY METRIC")
    for metric, (strat, val) in winners.items():
        print(f"  {metric:<25} → {strat} ({val:.1%})")

    # Save comparison
    with open("data/eval/strategy_comparison.json", "w") as f:
        # Remove result details to keep it compact
        compact = {s: {k: v for k, v in d.items() if k != "by_category"} for s, d in comparison.items()}
        json.dump(compact, f, indent=2)
    print("\n  ✓ Comparison saved to data/eval/strategy_comparison.json")

    # ------------------------------------------------------------------ #
    print("\n[4/4] Spot-check: hardest questions")
    print("-" * 60)
    all_results = report["results"]
    # Sort by correctness ascending — find the hardest ones
    sorted_by_correctness = sorted(all_results, key=lambda r: r["correctness"]["score"])
    print("  Lowest-scoring questions:")
    for r in sorted_by_correctness[:3]:
        print(f"\n  Q: {r['question']}")
        print(f"     Category    : {r['category']} / {r['difficulty']}")
        print(f"     Correctness : {r['correctness']['score']:.0%}")
        print(f"     Faithfulness: {r['faithfulness']['score']:.0%}")
        print(f"     Generated   : {r['generated_answer'][:100]}...")
        print(f"     Golden      : {r['golden_answer'][:100]}")

    print("\n" + "=" * 60)
    print("✅ PHASE 4 COMPLETE — Evaluation Framework")
    print("=" * 60)

    ov = report["overall"]
    print(f"""
Portfolio Numbers (use these in interviews):
  ✓ Evaluated on {len(questions)}-question golden dataset
  ✓ Answer Correctness  : {ov['answer_correctness']:.0%}
  ✓ Faithfulness        : {ov['faithfulness']:.0%}
  ✓ Retrieval Relevance : {ov['retrieval_relevance']:.0%}
  ✓ Citation Accuracy   : {ov['citation_accuracy']:.0%}

Next → Phase 5: FastAPI + Streamlit + Docker
  Run: .\\venv\\Scripts\\python.exe -m uvicorn src.api:app --reload
""")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
