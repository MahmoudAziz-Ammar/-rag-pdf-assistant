import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_API_TOKEN:
    raise ValueError("Définis HUGGINGFACEHUB_API_TOKEN dans le fichier .env")

def query_huggingface(prompt, model="google/flan-t5-small"):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.5
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data[0]['generated_text']


