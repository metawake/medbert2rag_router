import os
import chromadb
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Ensure dependencies are installed
os.system("pip install chromadb torch transformers pandas")

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="biomedical_faq")

# Load Hugging Face BioBERT Model (CPU-Only)
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to("cpu")  # Force CPU usage

# Load Sample RAG Data
csv_file = "biomedical_faqs.csv"
data = pd.DataFrame({
    "question": [
        "What is COVID-19?",
        "What are the symptoms of flu?",
        "How does ibuprofen work?"
    ],
    "answer": [
        "COVID-19 is a disease caused by the SARS-CoV-2 virus.",
        "Symptoms include fever, cough, and fatigue.",
        "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID)..."
    ]
})
data.to_csv(csv_file, index=False)

# Insert Data into ChromaDB
for idx, row in data.iterrows():
    collection.add(
        ids=[str(idx)],
        documents=[row["answer"]],
        metadatas=[{"question": row["question"]}]
    )

# Define a Query Function
def query_rag(query_text):
    results = collection.query(query_texts=[query_text], n_results=1)
    if results["documents"]:
        return results["documents"][0]
    else:
        return query_medbert(query_text)

# Define a function to query MedBERT
def query_medbert(query_text):
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return "Generated response from MedBERT (Placeholder)"

# Run a Sample Query
if __name__ == "__main__":
    print("Testing RAG search with ChromaDB and MedBERT fallback:")
    user_query = "What is COVID-19?"
    response = query_rag(user_query)
    print(f"User Query: {user_query}\nResponse: {response}")


    print("Testing MedBERT fallback:")
    user_query = "What is arthrosis, and its symptoms and treatment?"
    response = query_rag(user_query)
    print(f"User Query: {user_query}\nResponse: {response}")
