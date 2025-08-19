from module.preprocess import load_and_split_documents
from module.embed_store import (
    create_documents,
    get_embedding_model,
    create_vector_store,
    save_vector_store
)


docs = load_and_split_documents()  


embedding_model = get_embedding_model()


vectorstore = create_vector_store(docs, embedding_model)


save_vector_store(vectorstore, "vectorstore")
