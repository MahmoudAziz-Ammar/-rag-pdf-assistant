from app.rag_pipeline import query_huggingface

if __name__ == "__main__":
    question = "Quels sont les compétences techniques de John W. Smith ?"
    response = query_huggingface(question)
    print("Réponse générée :")
    print(response)
