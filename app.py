import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from rank_bm25 import BM25Okapi

# -----------------------------
# CONFIG
# -----------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key="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")

st.set_page_config(page_title="Hybrid Citation-Aware RAG", layout="wide")
st.title("📚 Hybrid Citation-Aware RAG (BM25 + FAISS)")
st.write("Semantic + Keyword Retrieval with Citations")

# -----------------------------
# LOAD INDEX
# -----------------------------
@st.cache_resource
def load_data():
    index = faiss.read_index("index/faiss.index")
    with open("index/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    documents = [doc["text"] for doc in metadata]
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return index, metadata, bm25

index, metadata, bm25 = load_data()

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Enter your question:")

if st.button("Ask") and query:

    try:
        # 1️⃣ FAISS (Dense Search)
        query_embedding = MODEL.encode([query])
        query_embedding = np.array(query_embedding)
        _, dense_indices = index.search(query_embedding, 3)

        # 2️⃣ BM25 (Keyword Search)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(bm25_scores)[-3:]

        # 3️⃣ Combine Results
        combined_indices = list(set(dense_indices[0].tolist() + sparse_indices.tolist()))

        retrieved_chunks = []
        citations = []

        for idx in combined_indices:
            chunk = metadata[idx]
            retrieved_chunks.append(chunk["text"])
            citations.append(chunk["page_number"])

        context = "\n\n".join(retrieved_chunks)

        # 4️⃣ LLM Prompt
        prompt = f"""
Answer using ONLY the context below.
Provide citations like (Page X).

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # Remove duplicate page numbers
        unique_pages = sorted(list(set(citations)))

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Sources:")
        st.write(", ".join([f"(Page {p})" for p in unique_pages]))

    except Exception as e:
        st.error(f"Error: {e}")


        #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