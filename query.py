import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=together_api_key,
    temperature=0.3,
    max_tokens=512
)

def rerank_with_bm25(query, docs, top_n=5):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    reranked_docs = [docs[i] for i in top_indices]
    
    return reranked_docs

def custom_retrieve(query):
    initial_docs = db.similarity_search(query, k=15)
    top_docs = rerank_with_bm25(query, initial_docs, top_n=5)
    return top_docs

class CustomRetriever(VectorStoreRetriever):
    def get_relevant_documents(self, query: str) -> list[Document]:
        return custom_retrieve(query)

retriever = CustomRetriever(vectorstore=db)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True, 
)

rephrase_prompt = PromptTemplate.from_template(
    "You are a helpful assistant that rewrites vague or unstructured insurance queries into clear, specific questions. "
    "Original Query: {user_query} \n\nRewritten Query:"
)

rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)

entity_prompt = PromptTemplate.from_template("""
Extract the following entities from the user query:

Query: "{user_query}"

Return JSON with keys:
- age (in years or "N/A")
- procedure (medical procedure or issue)
- location (city or "N/A")
- policy_duration (e.g., "2 months", or "N/A")

Return in JSON format only.
""")
entity_extraction_chain = LLMChain(llm=llm, prompt=entity_prompt)


raw_query = "Will dengue be covered?"  
rephrased_query = rephrase_chain.run({"user_query": raw_query})
entity_json = entity_extraction_chain.run({"user_query": raw_query})

formatted_prompt = f"""
You are an insurance claim assistant. Based on the retrieved clauses and the user query below, make a clear decision in valid JSON.

🟢 TASK:
Answer the following query. Clearly state:
- Whether the claim is Approved or Denied
- The reason using **specific clause numbers or text** from the policy
- Payout eligibility details

🧠 ENTITIES PARSED:
{entity_json}

🟡 QUERY:
"{rephrased_query}"

📝 FORMAT:
{{
  "decision": "Approved/Denied",
  "reason": "Your explanation here with clause reference(s)",
  "payout": "Coverage limit or amount (if mentioned)"
}}

⚠️ STRICT INSTRUCTIONS:
- Only output **valid JSON**
- Use exact clause text if possible
- Be consistent and clear

Answer below:
"""


print("▶️ Final processed query:", rephrased_query)
print("📊 Parsed Entities:", entity_json)

result = qa_chain.invoke({"query": formatted_prompt})
answer = result['result']
source_docs = result['source_documents']

print("\n🟩 Final Response:")
print(answer)

print("\n📚 Supporting Clauses Used:")
for i, doc in enumerate(source_docs):
    print(f"\n--- Clause {i+1} ---")
    print(doc.page_content.strip())
