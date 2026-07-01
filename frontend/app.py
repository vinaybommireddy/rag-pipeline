import streamlit as st
import httpx

API = "http://localhost:8000"

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("RAG Pipeline — Hybrid Search")

tab1, tab2, tab3 = st.tabs(["Ask", "Ingest", "Documents"])

with tab1:
    q = st.text_input("Ask a question")
    verify = st.checkbox("Verify citations (slower)")
    if st.button("Search") and q:
        with st.spinner("Retrieving..."):
            r = httpx.post(
                f"{API}/v1/ask",
                json={"question": q, "verify": verify},
                timeout=60
            )
        if r.status_code == 200:
            data = r.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.metric("Confidence", data["confidence"])
            st.subheader("Sources")
            for c in data.get("chunks", []):
                with st.expander(f"[{c['index']}] {c['source']}"):
                    st.write(c["text"])
            if verify and "citation_verification" in data:
                st.subheader("Citation Verification")
                for v in data["citation_verification"]:
                    emoji = "✅" if "SUPPORTED" in v["verdict"] else "❌"
                    st.write(f"{emoji} [{v['citation']}]: {v['verdict']}")
        else:
            st.error(r.text)

with tab2:
    strategy = st.selectbox("Chunking Strategy", ["fixed", "recursive"])
    if st.button("Ingest Documents"):
        with st.spinner("Ingesting..."):
            r = httpx.post(
                f"{API}/v1/ingest",
                json={"strategy": strategy},
                timeout=120
            )
        if r.status_code == 200:
            st.success(f"Done: {r.json()}")
        else:
            st.error(r.text)

with tab3:
    if st.button("List Documents"):
        r = httpx.get(f"{API}/v1/documents", timeout=30)
        if r.status_code == 200:
            data = r.json()
            st.metric("Total Chunks", data["total_chunks"])
            st.write("Sources:")
            for s in data["sources"]:
                st.write(f"- {s}")
        else:
            st.error(r.text)