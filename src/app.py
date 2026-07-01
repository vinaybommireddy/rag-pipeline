import streamlit as st
import requests
import pandas as pd
import json

# Page Config
st.set_page_config(
    page_title="Production Hybrid RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    h1 {
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        color: white;
        border: none;
        padding: 8px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.title("🤖 Enterprise RAG Pipeline Dashboard")
st.caption("Hybrid Retrieval-Augmented Generation Engine with Confidence Scoring & Citation Verification")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline Management")
    
    # Ingestion Component
    st.subheader("📁 Ingest New Document")
    uploaded_file = st.file_uploader("Upload a text document (.txt)", type=["txt"])
    if uploaded_file is not None:
        if st.button("🚀 Process & Index"):
            with st.spinner("Indexing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                try:
                    response = requests.post(f"{API_URL}/v1/ingest", files=files)
                    if response.status_code == 200:
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
                    
    # Document List
    st.subheader("📚 Indexed Documents")
    if st.button("🔄 Refresh Document List"):
        try:
            res = requests.get(f"{API_URL}/v1/documents")
            if res.status_code == 200:
                docs = res.json()
                if docs:
                    df = pd.DataFrame(docs)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No documents indexed yet.")
        except Exception as e:
            st.error(f"Could not load documents: {e}")

# Main Chat Interface
query = st.text_input("💬 Ask the RAG Pipeline a question:", placeholder="e.g., What is Python used for?")
top_k = st.slider("Select Retrieval Count (Top-K)", min_value=1, max_value=10, value=3)

if query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/v1/ask",
                json={"question": query, "top_k": top_k}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Confidence Score Metric UI
                conf = result.get("confidence", {})
                trustworthy = conf.get("trustworthy", True)
                score = conf.get("composite_score", 0.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Composite Score", f"{score:.1%}")
                with col2:
                    st.metric("Retrieval Confidence", f"{conf.get('retrieval_confidence', 0.0):.1%}")
                with col3:
                    st.metric("Citation Coverage", f"{conf.get('citation_coverage', 0.0):.1%}")
                
                if not trustworthy or result.get("abstained"):
                    st.warning(f"⚠️ IDK Guard: {conf.get('reason', 'Confidence threshold not met.')}")
                    
                # Answer box
                st.subheader("🤖 Generated Grounded Answer")
                st.info(result.get("answer"))
                
                # Citations
                st.subheader("📌 Citations Found")
                st.write(result.get("citations"))
                
                # Retrieved chunks
                st.subheader("🔍 Retrieved Context Chunks")
                for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                    with st.expander(f"Chunk [{i+1}] - Source: {chunk.get('metadata', {}).get('source')}"):
                        st.write(chunk.get("text"))
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI server. Make sure it is running at {API_URL}. Details: {e}")