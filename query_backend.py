import os
import re
import json
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import Field
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from rank_bm25 import BM25Okapi

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=together_api_key,
    temperature=0.3,
    max_tokens=1024,
)

def rerank_with_bm25(query, docs, top_n=5):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [docs[i] for i in top_indices]

def expand_with_neighbors(doc, window_size=1):
    """Pull previous and next chunks for full context (if chunk metadata present)."""
    expanded_texts = []
    if 'chunk_id' in doc.metadata and 'source' in doc.metadata:
        try:
            base_index = int(doc.metadata['chunk_id'])
            source = doc.metadata['source']
            for i in range(base_index - window_size, base_index + window_size + 1):
                chunk_key = f"{source}::chunk_{i}"
                if chunk_key in db.docstore._dict:
                    expanded_texts.append(db.docstore._dict[chunk_key].page_content)
        except Exception:
            
            return doc.page_content
    return "\n\n".join(expanded_texts) if expanded_texts else doc.page_content

def custom_retrieve(query, entities):
    initial_docs = db.similarity_search(query, k=20)

    
    synonyms_map = {
        "knee surgery": ["orthopedic surgery", "joint replacement", "ligament repair", "orthopedic procedures"],
        "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"]
        
    }

    boosted_docs = []
    for doc in initial_docs:
        score_boost = 0
        content_lower = doc.page_content.lower()

    
        procedure = entities.get("procedure", "").lower()
        if procedure:
            if procedure in content_lower:
                score_boost += 2
            for syn in synonyms_map.get(procedure, []):
                if syn in content_lower:
                    score_boost += 2

        if entities.get("location", "").lower() in content_lower:
            score_boost += 1
        boosted_docs.append((doc, score_boost))

    boosted_docs.sort(key=lambda x: x[1], reverse=True)
    sorted_docs = [doc for doc, _ in boosted_docs] or initial_docs

    
    top_docs = rerank_with_bm25(query, sorted_docs, top_n=5)

    expanded_docs = []
    for doc in top_docs:
        doc.page_content = expand_with_neighbors(doc, window_size=1)
        expanded_docs.append(doc)

    return expanded_docs

class CustomRetriever(VectorStoreRetriever):
    entities: Dict[str, Any] = Field(default_factory=dict)

    def get_relevant_documents(self, query: str):
        return custom_retrieve(query, self.entities)

rephrase_prompt = PromptTemplate.from_template(
    "You are a helpful assistant that rewrites vague or unstructured insurance queries into clear, specific questions. "
    "Original Query: {user_query} \n\nRewritten Query:"
)
rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)

entity_prompt = PromptTemplate.from_template("""
Extract the following structured fields from the user's query. Return JSON only.

Query: "{user_query}"

Return JSON with:
- age (e.g. "35" or "N/A")
- procedure (e.g. "hernia surgery", "eye treatment")
- location (e.g. "Mumbai", "N/A")
- duration (policy duration mentioned, e.g. "20 days", "2 months", "N/A")
""")
entity_extraction_chain = LLMChain(llm=llm, prompt=entity_prompt)


def process_user_query(raw_query, db):
    
    

    """
    1) Rephrase user query
    2) Extract entities
    3) Build entity-aware retriever
    4) Retrieve expanded clause paragraphs
    5) Build big decision prompt (unchanged)
    6) Invoke QA chain and parse JSON
    """
    try:
        
        rephrased_query = rephrase_chain.run({"user_query": raw_query})

        
        entity_json_raw = entity_extraction_chain.run({"user_query": rephrased_query})
        try:
            entity_data = json.loads(entity_json_raw)
            if not isinstance(entity_data, dict):
                entity_data = {}
        except Exception:
            
            entity_data = {}
            try:
                age_match = re.search(r'\"?age\"?\s*[:\-]\s*\"?(\d{1,3})', entity_json_raw, re.IGNORECASE)
                if age_match:
                    entity_data["age"] = age_match.group(1)
            except Exception:
                pass

        retriever = CustomRetriever(vectorstore=db, entities=entity_data)

        retrieved_docs = retriever.get_relevant_documents(rephrased_query)
        retrieved_clause_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        formatted_prompt = f"""
       You are an intelligent and highly accurate **Insurance Claim Decision Assistant**.

Your role is to carefully read insurance clauses and make clear, logical, and **fully traceable** decisions regarding claim queries.  
You must explicitly state whether the claim is **valid** (Approved) or **invalid** (Denied), and support your conclusion using **only the clauses provided**.

---

## PRIMARY OBJECTIVE:
Evaluate whether the user’s claim is eligible under the given policy clauses and user query details.  
Your reasoning must be:
- **Clause-based** (no guessing or external facts)
- **Traceable** (point to exact clauses)
- **Structured** (machine-readable JSON)

---

## TASK BREAKDOWN:
1. **Parse & Structure** — Identify the important elements from the query:
   - Age
   - Procedure
   - Location
   - Policy Duration

2. **Clause Analysis** — Go through each clause and determine:
   - If it **supports**, **rejects**, or is **neutral** towards the claim.
   - If the clause has **coverage limits** or payout amounts.

3. **Decision Logic**:
   - Approve if the clauses collectively allow coverage based on provided details.
   - Deny if any clause explicitly excludes or disqualifies the claim.
   - If unclear, default to Denied but explain the ambiguity.

4. **Justification**:
   - Quote the **exact clause text** and reference (e.g., "Clause 3(a)", "Section 5.2").
   - Ensure the user can **verify your answer** directly from the policy.

5. **Payout Extraction**:
   - If applicable, extract the payout condition or coverage amount exactly as written.
   - If not found, return `"Not specified"`.

---

## USER QUERY DETAILS:
- Age: {entity_data.get("age", "Unknown")}
- Procedure: {entity_data.get("procedure", "Unknown")}
- Location: {entity_data.get("location", "Unknown")}
- Policy Duration: {entity_data.get("duration", "Unknown")}
- Original Query: "{rephrased_query}"

---

## CLAUSES FOR EVALUATION:
You must base your decision **only** on the following clauses:  
{retrieved_clause_text}

---

## CLAUSE EVALUATION FORMAT:
For each clause, return:
- `"clause_reference"`: Exact reference if available (e.g., "Clause 3(a)"), else `"Not specified"`.
- `"clause_text"`: Exact quoted text from the clause.
- `"clause_summary"`: Simple restatement of what it means.
- `"clause_effect"`: One of `"Supports claim"`, `"Rejects claim"`, `"Neutral"`.
- `"payout_info"`: Exact payout/coverage terms if mentioned, else `"Not specified"`.

---

📝 OUTPUT FORMAT :
Return only valid JSON with the following keys:
{{
  "decision": "Approved or Denied",
  "reason": "Your explanation with clause references (quote exact lines if possible), also mention which clauses or which part of the text suggest this, to build user confidence in your answer. eg :- \"This is stated in clauses3(a) of particular policy\"",
  "payout": "Numeric amount or condition for coverage (e.g. 'Up to ₹50,000 per year'), or 'Not specified'"
}}
## STRICT RULES:
- No hallucinated clauses or numbers.
- Always match procedures with equivalent medical categories (e.g., "knee surgery" → "orthopedic procedures").
- Do not invent facts; use only what is in the clauses.
- Explicitly check **Age**, **Procedure**, **Location**, and **Policy Duration** against the clauses.
- Always include at least one clause_reference in your reasoning.

---

Answer below:

"""

    
        result = qa_chain.invoke({"query": formatted_prompt})
        raw_answer = result.get('result', "")
        sources = result.get('source_documents', [])
        source_clauses = [doc.page_content for doc in sources]

        
        try:
            json_string = re.search(r'\{.*\}', raw_answer, re.DOTALL).group()
            parsed_json = json.loads(json_string)
        except Exception:
            parsed_json = {
                "error": "Failed to parse JSON",
                "raw_output": raw_answer
            }

        return {
            "success": True,
            "rephrased_query": rephrased_query,
            "structured_entities": entity_data,
            "response_json": parsed_json,
            "clauses": source_clauses
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
