import os
import pdfplumber
from typing import List
from langchain_core.documents import Document

def preprocess_user_query(raw_query: str) -> str:
    raw_query = raw_query.lower()

    # Example mappings
    if "knee surgery" in raw_query:
        if "3-month" in raw_query or "three month" in raw_query:
            return "Is knee surgery covered under a 3-month policy?"
        elif "6-month" in raw_query or "six month" in raw_query:
            return "Is knee surgery covered under a 6-month policy?"
        else:
            return "Is knee surgery covered under the policy?"

    if "cataract" in raw_query:
        return "Is cataract surgery covered in the policy?"

    if "age" in raw_query or "years" in raw_query:
        return "What is the age limit for the policy?"

    if "pune" in raw_query:
        return "Are hospitals in Pune covered under this policy?"

    # Default fallback
    return raw_query


def load_pdfs(folder_path: str) -> List[dict]:
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        data.append({
                            "source": file,
                            "page": i + 1,
                            "text": text.strip()
                        })
    return data


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_and_split_documents(folder_path: str = "data") -> List[Document]:
    raw_docs = load_pdfs(folder_path)
    documents = []

    for doc in raw_docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            metadata = {
                "source": doc["source"],
                "page": doc["page"]
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    return documents
