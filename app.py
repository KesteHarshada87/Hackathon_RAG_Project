import streamlit as st
import faiss
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from groq import Groq
from rank_bm25 import BM25Okapi

# -----------------------------
# CONFIG
# -----------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key="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")
TOP_K = 5

st.set_page_config(page_title="Hybrid Citation-Aware RAG", layout="wide")
st.title("📚 Hybrid Citation-Aware RAG")
st.write("Semantic + Keyword Retrieval with Clean, Context-Only Citations")

# -----------------------------
# LOAD INDEX
# -----------------------------
@st.cache_resource
def load_data():
    index = faiss.read_index("index/faiss.index")
    with open("index/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    embeddings = np.load("index/embeddings.npy")
    with open("index/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return index, metadata, embeddings, bm25

index, metadata, embeddings, bm25 = load_data()

# -----------------------------
# QUERY INPUT
# -----------------------------
query = st.text_input("Enter your question:")

# -----------------------------
# RETRIEVAL (Hybrid + Filtering)
# -----------------------------
def retrieve_chunks(query, index, metadata, embeddings, bm25, top_k=TOP_K):
    query_vec = MODEL.encode([query])
    query_vec = np.array(query_vec)
    
    # Dense search
    _, dense_indices = index.search(query_vec, top_k)
    
    # Sparse search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[-top_k:]
    
    # Combine indices
    combined_indices = list(set(dense_indices[0].tolist() + sparse_indices.tolist()))
    
    # Rerank by combined score
    scored = []
    for idx in combined_indices:
        dense_score = -np.linalg.norm(query_vec - embeddings[idx])
        sparse_score = bm25_scores[idx]
        scored.append((dense_score + sparse_score, idx))
    scored.sort(reverse=True)
    
    # Select top_k
    top_indices = [idx for _, idx in scored[:top_k]]
    
    # Filter chunks containing query keywords and remove irrelevant sections
    filtered_chunks = []
    for idx in top_indices:
        chunk = metadata[idx]
        chunk_text_lower = chunk["text"].lower()
        section_lower = chunk["section"].lower()
        # Must contain at least one query word
        if not any(word in chunk_text_lower for word in tokenized_query):
            continue
        # Exclude irrelevant sections and skip chunks missing section/page
        if any(x in section_lower for x in ["summary", "personal application", "nces", "doi", "http", "references"]):
            continue
        if not chunk["section"] or not chunk["page_number"]:
            continue
        filtered_chunks.append(chunk)
    
    return filtered_chunks

# -----------------------------
# ANSWER GENERATION WITH CONTEXT-ONLY ENFORCEMENT
# -----------------------------
def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "Not found in the provided textbook.", {"sections": [], "pages": []}

    # Assemble context
    context_parts = [chunk["text"] for chunk in retrieved_chunks]

    # Map section -> page for inline citations
    inline_citations_map = {chunk["section"]: chunk["page_number"] for chunk in retrieved_chunks}

    context = "\n\n".join(context_parts)

    # Strict prompt: only use context
    prompt = f"""
Answer the question ONLY using the context below. 
Do NOT use any outside knowledge or references.
If the answer is not contained in the context, respond with: "Not found in the provided textbook."
Include inline citations in the format: (Section, Page Number) wherever relevant.

Context:
{context}

Question:
{query}
"""

    # Call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = response.choices[0].message.content

    # Replace placeholders with descriptive section + page
    for section, page in inline_citations_map.items():
        answer = answer.replace(f"(Section, Page Number)", f"({section}, {page})")

    # Build references JSON
    sections = sorted(list(set([chunk["section"] for chunk in retrieved_chunks])))
    pages = sorted(list(set([chunk["page_number"] for chunk in retrieved_chunks])))
    references = {"sections": sections, "pages": pages}

    return answer, references

# -----------------------------
# STREAMLIT UI
# -----------------------------
if st.button("Ask") and query:
    retrieved_chunks = retrieve_chunks(query, index, metadata, embeddings, bm25)
    answer, references = generate_answer(query, retrieved_chunks)

    st.subheader("Answer:")
    st.write(answer)

    st.subheader("References:")
    st.write(json.dumps(references, indent=2))