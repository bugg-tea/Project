import os
import glob
import nltk
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import your existing loaders
from module.pdf_loader import load_pdfs

from module.file_loader import load_uploaded_files  # can handle docx, emails, txt, etc.

nltk.download("punkt")

# Load HuggingFace token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Constants
PDF_FOLDER_PATH = "data"
VECTOR_STORE_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --------------------------
# 🔹 Load default PDFs from data/
# --------------------------
def load_all_documents(folder_path: str) -> List[Document]:
    all_docs = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    for file_path in pdf_files:
        docs = load_pdfs(file_path)  # using your custom PDF loader
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        all_docs.extend(docs)

    return all_docs


# --------------------------
# 🔹 Load uploaded files (any format your loader supports)
# --------------------------
def load_uploaded_files(file_paths: List[str]) -> List[Document]:
    all_docs = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            docs = load_pdfs(file_path)
        else:
            docs = load_uploaded_files(file_path)  # your generic loader

        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        all_docs.extend(docs)

    return all_docs


# --------------------------
# 🔹 Sentence-aware chunking
# --------------------------
def split_and_tag_chunks(docs: List[Document]) -> List[Document]:
    chunks = []
    chunk_id = 0
    for doc in docs:
        sentences = nltk.sent_tokenize(doc.page_content)
        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) < CHUNK_SIZE:
                buffer += " " + sentence
            else:
                chunks.append(Document(
                    page_content=buffer.strip(),
                    metadata={
                        "source": doc.metadata["source"],
                        "chunk_id": chunk_id
                    }
                ))
                chunk_id += 1
                buffer = sentence
        if buffer.strip():
            chunks.append(Document(
                page_content=buffer.strip(),
                metadata={
                    "source": doc.metadata["source"],
                    "chunk_id": chunk_id
                }
            ))
            chunk_id += 1

    return chunks


# --------------------------
# 🔹 Embedding model
# --------------------------
def get_embedding_model(model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


# --------------------------
# 🔹 Create vector store
# --------------------------
def create_vector_store(documents: List[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(documents, embedding_model)


# --------------------------
# 🔹 Save & load vector store
# --------------------------
def save_vector_store(vectorstore: FAISS, path: str = VECTOR_STORE_PATH):
    vectorstore.save_local(folder_path=path)


def load_default_vectorstore():
    embedding_model = get_embedding_model()
    return FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)


def build_vectorstore_from_files(file_paths: List[str]):
    docs = load_uploaded_files(file_paths)
    chunks = split_and_tag_chunks(docs)
    embedding_model = get_embedding_model()
    return create_vector_store(chunks, embedding_model)


# --------------------------
# ✅ Main execution (default build from data/)
# --------------------------
if __name__ == "__main__":
    print("📄 Loading all PDFs from data/ folder...")
    raw_docs = load_all_documents(PDF_FOLDER_PATH)

    print(f"📑 Loaded {len(raw_docs)} pages. Splitting and tagging...")
    chunks = split_and_tag_chunks(raw_docs)

    print(f"🔍 Embedding and creating FAISS index for {len(chunks)} chunks...")
    embedding_model = get_embedding_model()
    db = create_vector_store(chunks, embedding_model)

    print("💾 Saving FAISS index to disk...")
    save_vector_store(db)

    print("✅ Vector store created and saved successfully.")
