from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import uvicorn

# Load .env variables
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
)

VECTOR_STORE_DIR = "../vectore_store"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

app = FastAPI(title="Member QA API")

class QARequest(BaseModel):
    question: str
    top_k: int = 5

def search_for_answer_llm(question: str, top_k: int = 10):
    query_embedding = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Compose context for LLM
    context = "\n".join([
        # Use chunked metadata if present, fallback to single-message
        f"users: {meta.get('users', meta.get('user_name', 'UNKNOWN'))}, msg: {doc}"
        for doc, meta in zip(docs, metas)
    ])
    if not context:
        answer = "No relevant information found."
    else:
        prompt = (
            f"Answer the question using only the following context.\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

    # Optionally return context and top hit info
    return {
        "answer": answer
    }

@app.post("/ask")
def ask(request: QARequest):
    return search_for_answer_llm(request.question, top_k=request.top_k)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default to 8000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)
