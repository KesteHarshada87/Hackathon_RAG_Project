import fitz
import uuid
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import re

# -----------------------------
# CONFIG
# -----------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHUNK_SIZE = 500  # characters
os.makedirs("index", exist_ok=True)

# -----------------------------
# SECTION-AWARE CHUNKING (IMPROVED)
# -----------------------------
def extract_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    doc = fitz.open(pdf_path)
    chunks = []
    current_section = "Introduction"

    # Regex patterns to detect section headings (from OpenStax formatting)
    section_pattern = re.compile(r'^(Chapter \d+: .*|[0-9]+(\.[0-9]+)* .+)$')  # e.g., "6.2 Classical Conditioning"

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text("text")

        # Split text by lines to detect section headings
        lines = text.splitlines()
        for line in lines:
            line_strip = line.strip()
            # Detect section headings based on pattern
            if section_pattern.match(line_strip):
                current_section = line_strip

        # Split page into chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size].strip()
            if not chunk_text:
                continue
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "page_number": page_number + 1,
                "section": current_section
            })

    return chunks

# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index(chunks):
    embeddings = MODEL.encode([c["text"] for c in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save FAISS index
    faiss.write_index(index, "index/faiss.index")
    # Save metadata and embeddings
    with open("index/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("index/embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    # Build BM25 index
    tokenized_docs = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    with open("index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("Index created successfully!")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    pdf_path = "data/book.pdf"

    if not os.path.exists("index/faiss.index"):
        print("Extracting chunks...")
        chunks = extract_chunks(pdf_path)
        print("Building index...")
        build_index(chunks)
        print("Done!")
    else:
        print("Index already exists. Skipping build.")