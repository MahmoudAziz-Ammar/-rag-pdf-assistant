import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, create_faiss_index, search
from llm_utils import generate_answer

# Nom du modèle SentenceTransformer
MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Assistant PDF RAG", page_icon="📄")
st.title("📚 Assistant intelligent sur document PDF")

uploaded_file = st.file_uploader("📤 Charge ton fichier PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Lecture et traitement du PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        st.success(f"{len(chunks)} passages extraits ✅")

        model = SentenceTransformer(MODEL_NAME)
        embeddings = embed_chunks(chunks, model)
        index = create_faiss_index(np.array(embeddings))

    st.subheader("🔍 Pose ta question sur le document")
    query = st.text_input("Ex : Quelle est la différence entre PCA et t-SNE ?")

    if query:
        with st.spinner("🤖 Génération de la réponse..."):
            results = search(query, index, chunks, model)
            context = "\n".join(results)
            answer = generate_answer(query, context)

        st.markdown("### 🧠 Réponse générée :")
        st.write(answer)

        st.markdown("### 📌 Passages utilisés pour générer la réponse :")
        for i, passage in enumerate(results):
            st.markdown(f"**Passage {i+1}** :\n\n{passage}\n\n---")
