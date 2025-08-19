# Insurance claim assistant (hackrx) – Document-based Question Answering System


---

## Project Overview
HackRx is a **document-based question answering system** designed to process **insurance policy documents** and answer queries in natural language.  
It leverages **LangChain**, **Together.ai LLM (Mixtral-8x7B)**, **FAISS vector store**, and **HuggingFace embeddings** for accurate, interpretable results.

---

##  Key Features
- Handles **vague, incomplete, or plain English queries**.  
- Performs **explicit entity extraction**: age, procedure, location, policy duration.  
- Provides **structured JSON output** including:
  - Decision  
  - Payout  
  - Clause-wise evaluation  
- Maintains **source clause traceability** for audit and interpretability.  
- Implements **prompt-based query rewriting** to improve relevance.  
- Integrates **BM25 reranking** for better retrieval accuracy.

---

##  Tech Stack
- **Language & Frameworks:** Python, LangChain  
- **Vector Database:** FAISS with HuggingFace embeddings  
- **LLM:** Together.ai Mixtral-8x7B  
- **Frontend:** Streamlit  
- **PDF Processing:** pdfplumber  

---

##  Usage

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000


