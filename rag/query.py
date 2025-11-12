import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "../vectore_store"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"

def semantic_search(query, top_k=5):
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query])[0].tolist()
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ids = results["ids"][0]

    for i, (doc, meta, dist, msg_id) in enumerate(zip(docs, metas, dists, ids)):
        print(f"\n#{i+1} | Message ID: {msg_id} | Score: {1-dist:.4f}")
        print(f"{doc}")
        print(f"Users: {meta['users']} | Timestamps: {meta['timestamps']}")
        print(f"Message IDs: {meta['messages']}")

if __name__ == "__main__":
    query = input("Enter your search/query: ")
    semantic_search(query, top_k=5)
