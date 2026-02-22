import fitz
import uuid
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "page_number": page_number + 1,
                "section": "Unknown"
            })

    return chunks


def build_index(chunks):
    embeddings = MODEL.encode([c["text"] for c in chunks])

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, "index/faiss.index")

    with open("index/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index created successfully!")

if __name__ == "__main__":
    pdf_path = "data/book.pdf"

    print("Extracting chunks...")
    chunks = extract_chunks(pdf_path)

    print("Building index...")
    build_index(chunks)

    print("Done!")