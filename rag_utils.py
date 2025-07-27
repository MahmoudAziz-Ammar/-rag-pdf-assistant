import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def extract_text_from_pdf(file) -> str:
    """Extrait tout le texte d’un fichier PDF"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Coupe le texte en morceaux pour faciliter la recherche.
    chunk_size : nombre de caractères par chunk
    overlap : recouvrement entre chunks pour garder du contexte
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # pour garder un peu de recouvrement
    return chunks

def embed_chunks(chunks, model):
    """Calcule les embeddings pour chaque chunk avec SentenceTransformer"""
    embeddings = model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    """
    Crée un index FAISS sur les embeddings pour recherche rapide.
    Utilisation d'un index flat L2 simple ici.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search(query, index, chunks, model, top_k=3):
    """
    Recherche les passages les plus proches de la requête.
    Retourne les passages correspondants.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results
