import os
import hashlib
import inspect
import streamlit as st
from typing import List

from module.file_loader import load_uploaded_files
from module.pdf_loader import load_and_chunk_pdfs
from module.embed_store import (
    load_all_documents,
    split_and_tag_chunks,
    get_embedding_model,
    create_vector_store,
    save_vector_store,
)

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document


from query_backend import process_user_query


PDF_FOLDER_PATH = "data"
SAVED_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

st.set_page_config(page_title="Insurance Claim Assistant", page_icon="💼", layout="wide")
st.title("💼 Insurance Claim Assistant")

st.sidebar.header("Data source")
data_option = st.sidebar.radio("Choose data:", ("Use sample dataset (data/)", "Upload my documents"))


def files_fingerprint(uploaded_files):
    m = hashlib.sha256()
    for f in uploaded_files:
        m.update(f.name.encode() + b"::" + f.getbuffer()[:1024])
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def _cached_embedding_model():
    return get_embedding_model()

def load_or_build_sample_index():
    try:
        emb = _cached_embedding_model()
        if os.path.exists(SAVED_INDEX_PATH) and os.listdir(SAVED_INDEX_PATH):
            st.info("Found existing FAISS index — attempting to load it (faster).")
            from langchain_community.vectorstores import FAISS
            try:
                db = FAISS.load_local(SAVED_INDEX_PATH, emb, allow_dangerous_deserialization=True)
                return db, "loaded"
            except Exception as e:
                st.warning(f"Failed to load saved index (will rebuild). Error: {e}")

        st.info("Loading PDFs from data/ and building vector index. This may take a while on first run.")
        raw_docs = load_all_documents(PDF_FOLDER_PATH)  
        chunks = split_and_tag_chunks(raw_docs)         
        emb = _cached_embedding_model()
        db = create_vector_store(chunks, emb)
        save_vector_store(db, path=SAVED_INDEX_PATH)
        return db, "built"
    except Exception as e:
        st.error(f"Error while building/loading sample index: {e}")
        raise


def build_index_from_uploaded_text(text: str):
    chunks = []
    i = 0
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        chunk_text = text[start:start + CHUNK_SIZE].strip()
        chunks.append(Document(page_content=chunk_text, metadata={"source": "uploaded", "chunk_id": i}))
        i += 1
        start += step

    emb = _cached_embedding_model()
    db = create_vector_store(chunks, emb)
    return db, chunks

db = None
chunks_preview: List[Document] = []

if data_option == "Use sample dataset (data/)":
    with st.spinner("Loading and processing sample dataset..."):
        try:
            db, status = load_or_build_sample_index()
            st.success(f"Sample dataset vectorstore ready ({status}).")
            
            
            try:
                sample_keys = list(db.docstore._dict.keys())[:3]
                st.markdown("### Sample indexed chunks (from FAISS docstore)")
                for k in sample_keys:
                    chunk = db.docstore._dict[k]
                    st.write(f"**{k}**")
                    st.write(chunk.page_content[:800].strip() + ("..." if len(chunk.page_content) > 800 else ""))
            except Exception:
                st.info("Preview not available for this FAISS instance.")
        except Exception as e:
            st.error(f"Could not prepare sample index: {e}")

else:
    st.info("Upload PDF / DOCX / EML / TXT files (multiple allowed).")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "eml", "txt"], accept_multiple_files=True)
    if uploaded_files:
        fingerprint = files_fingerprint(uploaded_files)
        st.write(f"Files uploaded — fingerprint `{fingerprint[:12]}...`")
        with st.spinner("Reading and processing uploaded files..."):
            try:
                all_text = load_uploaded_files(uploaded_files)
                if not all_text.strip():
                    st.error("No readable text was extracted from uploaded files.")
                else:
                    db, chunks_preview = build_index_from_uploaded_text(all_text)
                    st.success(f"Created vectorstore from uploads (chunks: {len(chunks_preview)}).")
                    st.markdown("### Sample processed chunks (first 3):")
                    for idx, c in enumerate(chunks_preview[:3]):
                        st.write(f"**Chunk {idx}** — metadata: {c.metadata}")
                        st.write(c.page_content[:800] + ("..." if len(c.page_content) > 800 else ""))
            except Exception as e:
                st.error(f"Failed to process uploaded files: {e}")


st.markdown("---")
st.subheader("Ask a question about the policy (example: 'I had hernia surgery 20 days after buying the policy — can I claim?')")
user_query = st.text_area("Enter your query here", height=120)

if st.button("Run Query"):

    if db is None:
        st.warning("Please load a dataset (sample or upload files) before running a query.")
    elif not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query with the QA engine..."):
            try:
                sig = inspect.signature(process_user_query)
                if len(sig.parameters) == 2:
                    result = process_user_query(user_query, db)
                else:
                    result = process_user_query(user_query)

                
                if isinstance(result, dict):
                    if result.get("success"):
                        st.success("✅ Query processed")
                        st.markdown("**Rephrased Query:**")
                        st.write(result.get("rephrased_query", "N/A"))

                        st.markdown("**Parsed Entities:**")
                        if "structured_entities" in result:
                            st.json(result["structured_entities"])
                        elif "rephrased_query" in result:
                            st.write("No structured entities returned by backend.")
                        else:
                            st.write(result.get("rephrased_query", "N/A"))

                        st.markdown("**Decision (model JSON)**")
                        st.json(result.get("response_json", {}))

                        st.markdown("**Top source clauses returned**")
                        for i, clause in enumerate(result.get("clauses", [])[:6]):
                            st.write(f"--- Clause {i+1} ---")
                            st.write(clause[:1000] + ("..." if len(clause) > 1000 else ""))

                    else:
                        st.error("❌ Backend failed")
                        st.text(result.get("error", "Unknown error"))
                else:
                    st.write(result)

            except Exception as e:
                st.error(f"Unexpected error while querying: {e}")
