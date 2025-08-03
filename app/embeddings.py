from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

def create_chroma_db(chunks, persist_directory="chroma_db"):
    # Initialise le modèle d'embedding HuggingFace
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Crée une base ChromaDB et y insère les chunks vectorisés
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    vectordb.persist()  # Sauvegarde les données localement
    print(f"✅ ChromaDB initialisée avec {len(chunks)} chunks")
    return vectordb

def load_chroma_db(persist_directory="chroma_db"):
    # Recharge une base ChromaDB existante
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectordb
