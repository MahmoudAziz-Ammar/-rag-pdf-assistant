import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer le token Hugging Face depuis la variable d'environnement
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Créer le client une fois
client = InferenceClient(model=LLM_MODEL, token=HUGGINGFACE_TOKEN)

def generate_answer(query, context):
    prompt = f"Contexte : {context}\nQuestion : {query}\nRéponse :"
    messages = [
        {"role": "system", "content": "Tu es un assistant utile."},
        {"role": "user", "content": prompt}
    ]
    # Utilise .generate() ou .call() sur client.chat, pas client.chat() directement
    response = client.chat.generate(messages=messages)
    # Le texte généré se trouve dans choices[0].message.content
    return response.choices[0].message.content

