import os
import tempfile
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

from email import policy
from email.parser import BytesParser

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    body = msg.get_body(preferencelist=('plain'))
    return body.get_content() if body else ""

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_uploaded_files(uploaded_files):
    """Reads and extracts text from uploaded files of different types."""
    all_text = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if ext == ".pdf":
            all_text += extract_text_from_pdf(tmp_path) + "\n"
        elif ext == ".docx":
            all_text += extract_text_from_docx(tmp_path) + "\n"
        elif ext in [".eml", ".msg"]:
            all_text += extract_text_from_eml(tmp_path) + "\n"
        elif ext == ".txt":
            all_text += extract_text_from_txt(tmp_path) + "\n"
        else:
            print(f"Unsupported file type: {uploaded_file.name}")

        os.remove(tmp_path)
    return all_text.strip()
