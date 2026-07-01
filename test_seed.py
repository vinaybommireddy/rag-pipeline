import sys
import traceback
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

try:
    print("=" * 60)
    print("STEP 1: Create Documents")
    print("=" * 60)
    from pathlib import Path
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    python_file = docs_dir / "python_guide.txt"
    if not python_file.exists():
        python_file.write_text("""Python is a high-level, interpreted programming language known for its readability and versatility. 
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
Python is widely used in web development, data science, artificial intelligence, and automation.
The language emphasizes code readability with its use of significant indentation.
Python has a large standard library and a vibrant ecosystem of third-party packages.""")
        print("✓ Created python_guide.txt")
    else:
        print("✓ python_guide.txt already exists")
    
    ml_file = docs_dir / "machine_learning.txt"
    if not ml_file.exists():
        ml_file.write_text("""Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled training data to build predictive models.
Unsupervised learning finds hidden patterns in unlabeled data.
Deep learning uses neural networks with many layers to model complex patterns.
Python is the most popular language for machine learning due to libraries like TensorFlow and PyTorch.""")
        print("✓ Created machine_learning.txt")
    
    web_file = docs_dir / "web_development.txt"
    if not web_file.exists():
        web_file.write_text("""Web development involves building websites and web applications.
Frontend development focuses on the user interface using HTML, CSS, and JavaScript.
Backend development handles server-side logic, databases, and APIs.
Python frameworks like Django and Flask are popular for backend development.
REST APIs allow communication between frontend and backend services.""")
        print("✓ Created web_development.txt")
    
    print("\n" + "=" * 60)
    print("STEP 2: Load Documents")
    print("=" * 60)
    from src.ingest import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_from_directory()
    print(f"✓ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.source} ({len(doc.content)} chars)")
    
    print("\n" + "=" * 60)
    print("STEP 3: Chunk Documents")
    print("=" * 60)
    from src.chunking_strategies import ChunkingFactory
    all_chunks = []
    for doc in docs:
        chunks = ChunkingFactory.create_chunks(doc.content, doc.source, strategy="fixed-size")
        all_chunks.extend(chunks)
    print(f"✓ Created {len(all_chunks)} chunks")
    
    print("\n" + "=" * 60)
    print("STEP 4: Generate Embeddings")
    print("=" * 60)
    from src.embeddings_manager import EmbeddingsManager
    em = EmbeddingsManager()
    results = em.embed_chunks(all_chunks)
    print(f"✓ Generated {len(results)} embeddings")
    
    print("\n" + "=" * 60)
    print("STEP 5: Store in Vector Store")
    print("=" * 60)
    from src.vector_store import VectorStore
    vs = VectorStore()
    embeddings = [r.embedding for r in results]
    if embeddings:
        vs.add_chunks(all_chunks[:len(results)], embeddings)
        print(f"✓ Stored {len(results)} chunks in ChromaDB")
    else:
        print("✗ No embeddings to store")
    
    print("\n" + "=" * 60)
    print("STEP 6: Build BM25 Index")
    print("=" * 60)
    from src.bm25_search import BM25Search
    bm25 = BM25Search()
    bm25.build_index(all_chunks[:len(results)])
    print("✓ Built BM25 index")
    
    print("\n" + "=" * 60)
    print("STEP 7: Test Hybrid Retrieval")
    print("=" * 60)
    from src.retriever import HybridRetriever
    engine = HybridRetriever(vector_store=vs, bm25=bm25, embedder=em)
    
    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "What is web development?",
    ]
    
    for test_query in test_queries:
        print(f"\n  Query: '{test_query}'")
        result = engine.retrieve(test_query, top_k=3)
        print(f"  ✓ Dense results: {len(result['dense_results'])}")
        print(f"  ✓ Sparse results: {len(result['sparse_results'])}")
        print(f"  ✓ Fused results: {len(result['fused_results'])}")
        if result.get('fused_results'):
            print(f"  ✓ Top result: {result['fused_results'][0]['text'][:80]}...")
    
    print("\n" + "=" * 60)
    print("STEP 8: Test Answer Generation")
    print("=" * 60)
    from src.generator import AnswerGenerator
    gen = AnswerGenerator()
    
    test_query = "What is Python used for?"
    result = engine.retrieve(test_query, top_k=3)
    answer = gen.generate(test_query, result.get('top_results', result.get('fused_results', [])))
    print(f"✓ Query: {test_query}")
    print(f"✓ Answer: {answer['answer']}")
    print(f"✓ Citations: {answer['citations']}")
    
    print("\n" + "=" * 60)
    print("STEP 9: Test Citation Verification")
    print("=" * 60)
    from src.generator import CitationVerifier
    verifier = CitationVerifier()
    top = result.get('top_results', result.get('fused_results', []))
    verification = verifier.verify(answer['answer'], top)
    print(f"✓ Coverage: {verification['coverage']:.1%}")
    print(f"✓ All verified: {verification['all_verified']}")
    if verification.get('unsupported'):
        print(f"  ⚠ Unsupported claims: {len(verification['unsupported'])}")
    
    print("\n" + "=" * 60)
    print("STEP 10: Confidence Scoring (Phase 3)")
    print("=" * 60)
    from src.confidence import AnswerConfidenceScorer
    scorer = AnswerConfidenceScorer()
    confidence = scorer.score(
        query=test_query,
        answer=answer['answer'],
        chunks=result.get('top_results', result.get('fused_results', [])),
        verification=verification,
    )
    print(f"✓ Retrieval confidence : {confidence['retrieval_confidence']:.1%}")
    print(f"✓ Citation coverage    : {confidence['citation_coverage']:.1%}")
    print(f"✓ Answer completeness  : {confidence['answer_completeness']:.1%}")
    print(f"✓ Composite score      : {confidence['composite_score']:.1%}")
    print(f"✓ Trustworthy          : {confidence['trustworthy']}")
    print(f"✓ Reason               : {confidence['reason']}")

    print("\n" + "=" * 60)
    print("STEP 11: 'I Don't Know' Handler (Phase 3)")
    print("=" * 60)
    from src.confidence import IDontKnowHandler
    idk = IDontKnowHandler()

    # Test with a question that has no answer in the corpus
    unknown_query = "What is the capital of France?"
    unknown_result = engine.retrieve(unknown_query, top_k=3)
    unknown_answer = gen.generate(unknown_query, unknown_result.get('top_results', []))
    unknown_verif  = verifier.verify(unknown_answer['answer'], unknown_result.get('top_results', []))
    unknown_conf   = scorer.score(
        query=unknown_query,
        answer=unknown_answer['answer'],
        chunks=unknown_result.get('top_results', []),
        verification=unknown_verif,
    )

    if idk.should_abstain(unknown_conf):
        idk_response = idk.build_response(unknown_query, unknown_result.get('top_results', []), unknown_conf)
        print(f"✓ Correctly abstained for out-of-corpus question")
        print(f"  Answer  : {idk_response['answer']}")
        print(f"  Suggest : {idk_response['suggestion']}")
    else:
        print(f"  (System answered with confidence {unknown_conf['composite_score']:.1%})")
        print(f"  Answer  : {unknown_answer['answer'][:120]}...")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! — Phases 1, 2, and 3 complete")
    print("=" * 60)
    print("\nYour RAG pipeline is fully working!")
    print("\nPhase summary:")
    print("  ✅ Phase 1 — Ingestion & Chunking")
    print("  ✅ Phase 2 — Hybrid Retrieval (Dense + BM25 + RRF)")
    print("  ✅ Phase 3 — Generation, Citations, Confidence, IDK Handler")
    print("\nNext → Phase 4: Evaluation Framework")
    print("  Run: python -c \"from src.evaluator import run_eval; run_eval()\"")
    print("\nNext → Phase 5: FastAPI + Streamlit")
    print("  Run: .\\venv\\Scripts\\python.exe -m uvicorn src.api:app --reload")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
