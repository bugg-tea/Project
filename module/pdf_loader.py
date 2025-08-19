import os
import pdfplumber
from typing import List, Dict

def load_pdfs(folder_path: str) -> List[Dict]:
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

def load_and_chunk_pdfs(folder_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    raw_docs = load_pdfs(folder_path)
    all_chunks = []

    for doc in raw_docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })
    
    return all_chunks
