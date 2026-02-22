import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
# Load embedding model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# 🔐 Hardcoded Groq API Key (Hackathon Use Only)
client = Groq(api_key="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")

st.set_page_config(page_title="Citation-Aware RAG", layout="wide")
st.title("📚 Citation-Aware RAG System (Groq Powered)")
st.write("Ask questions from Psychology 2e and get answers with citations.")

# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
@st.cache_resource
def load_index():
    index = faiss.read_index("index/faiss.index")
    with open("index/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_index()


# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Enter your question:")

if st.button("Ask") and query:

    try:
        # 1️⃣ Embed Query
        query_embedding = MODEL.encode([query])
        query_embedding = np.array(query_embedding)

        # 2️⃣ Search FAISS
        k = 5
        distances, indices = index.search(query_embedding, k)

        retrieved_chunks = []
        citations = []

        for idx in indices[0]:
            chunk = metadata[idx]
            retrieved_chunks.append(chunk["text"])
            citations.append(f"(Page {chunk['page_number']})")

        context = "\n\n".join(retrieved_chunks)

        # 3️⃣ Create Prompt
        prompt = f"""
Answer the question using ONLY the context below.
Provide citations like (Page X).

Context:
{context}

Question:
{query}
"""

        # 4️⃣ Call Groq Model (Latest Working Model)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # 5️⃣ Display
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Sources:")
        for c in citations:
            st.write(c)

    except Exception as e:
        st.error(f"Error: {e}")