from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_retriever(persist_directory="chroma_db", top_k=5):
    # Charger la base ChromaDB avec le modèle d'embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # Créer un retriever pour interroger la base
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    return retriever

def retrieve_documents(query, retriever):
    # Récupérer les documents pertinents en fonction de la requête
    docs = retriever.get_relevant_documents(query)
    return docs
