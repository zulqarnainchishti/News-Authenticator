import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/articles.parquet"
EMB_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/embeddings.npy"
INDEX_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def format_block(row):
    cats = ", ".join(row.get("categories", []))
    ents = ", ".join(row.get("entities", []))
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()

    block = (
        f"Categories: {cats}.\n"
        f"Entities: {ents}.\n"
        f"Title: {title}.\n"
        f"Content: {text}"
    )
    
    return block


def build_index(embeddings):
    
    # for cosine similarity
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    
    # inner Product = cosine similarity on normalized vectors
    index = faiss.IndexFlatIP(d)
    index.add(embeddings) 
    
    return index


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    print(f"Loaded {len(df)} records")

    blocks = [format_block(row) for _, row in df.iterrows()]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(blocks, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    index = build_index(embeddings)

    np.save(EMB_PATH, embeddings)
    print(f"Embeddings saved to: {EMB_PATH}")

    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to: {INDEX_PATH}")