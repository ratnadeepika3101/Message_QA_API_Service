from fastapi import FastAPI, Query
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import uvicorn
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
)


# --- RAG retrieval setup ---
VECTOR_STORE_DIR = "../vectore_store"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"
def local_rag_query(question, top_k=5):
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    query_embedding = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Robust user attribution for both chunked and single-message metadata
    context = "\n".join([
        f"users: {meta.get('users', meta.get('user_name', 'UNKNOWN'))}, msg: {doc}"
        for doc, meta in zip(docs, metas)
    ])
    print("\n---- Retrieved Context ----")
    print(context)

    if not context:
        return "No relevant context found."
    
    prompt = (
        f"Answer the question using only the following context.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    print("\n---- Prompt to LLM ----")
    print(prompt)

    response = llm.invoke(prompt)
    print("\n---- LLM Answer ----")
    print(response.content if hasattr(response, "content") else str(response))
    return response.content if hasattr(response, "content") else str(response)

if __name__ == "__main__":
    q = "What does Fatima want?"
    local_rag_query(q, top_k=10)