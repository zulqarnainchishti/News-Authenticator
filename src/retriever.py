import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/articles.parquet"
EMB_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/embeddings.npy"
INDEX_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

df = None
embeddings = None
index = None
model = None

def initialize_resources():
    print("Loading retriever resources...")
    global df, embeddings, index, model
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    embeddings = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    print("Retriever resources loaded.")

def similar_articles(query, k = 5):
    # Encode and normalize for cosine similarity
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # Search
    distances, indices = index.search(query_emb, k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        record = df.iloc[idx]
        results.append({
            "score": float(score),
            "published": record.get("published", ""),
            "categories": record.get("categories", []),
            "entities": record.get("entities", []),
            "title": record.get("title", ""),
            "content": record.get("content", "")
        })

    return results