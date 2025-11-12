import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# SETTINGS
DATA_DIR = "../messages"  # Path to your messages folder in root
VECTOR_STORE_DIR = "../vectore_store"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 5  # You can tune this (try 3–8 depending on your data)

def load_all_messages(data_dir):
    all_messages = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                all_messages.extend(json.load(f))
    return all_messages

def chunk_messages(messages, chunk_size):
    chunks = []
    chunk_metadatas = []
    chunk_ids = []
    for i in range(0, len(messages) - chunk_size + 1):
        chunk_lines = [
            f"user: {msg['user_name']}, msg: {msg['message']}"
            for msg in messages[i:i+chunk_size]
        ]
        chunk_text = "\n".join(chunk_lines)
        # ChromaDB requires metadata values to be primitive types, not lists
        chunk_meta = {
            "messages": "|".join([msg["id"] for msg in messages[i:i+chunk_size]]),
            "users": "|".join([msg["user_name"] for msg in messages[i:i+chunk_size]]),
            "timestamps": "|".join([msg["timestamp"] for msg in messages[i:i+chunk_size]]),
        }
        chunks.append(chunk_text)
        chunk_metadatas.append(chunk_meta)
        chunk_ids.append(f"chunk_{i}")
    return chunks, chunk_metadatas, chunk_ids

def main():
    print(f"Loading messages from: {DATA_DIR}")
    messages = load_all_messages(DATA_DIR)
    print(f"Loaded {len(messages)} messages.")

    print(f"Creating chunks of size {CHUNK_SIZE} ...")
    texts, metadatas, ids = chunk_messages(messages, CHUNK_SIZE)
    print(f"Created {len(texts)} context chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

    print(f"Initializing ChromaDB at {VECTOR_STORE_DIR}")
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Adding embeddings to ChromaDB...")
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist()
    )
    print(f"Successfully stored {len(texts)} chunks in ChromaDB ({COLLECTION_NAME})")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()
