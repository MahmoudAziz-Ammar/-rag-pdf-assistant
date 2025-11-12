from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Charger ton token Hugging Face (optionnel si tu télécharges le modèle localement)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Charger un modèle d'embedding gratuit depuis Hugging Face
# (ce modèle est petit, rapide et très bon pour le texte français/anglais)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # taille des embeddings pour ce modèle

# Charger le modèle
model = SentenceTransformer(EMBED_MODEL, use_auth_token=HF_TOKEN)

# Créer ton splitter pour découper les textes
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    """Lit un PDF et le découpe en petits morceaux de texte"""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Crée les embeddings à partir des textes"""
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings
