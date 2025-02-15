import os
import chromadb
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from rdflib import Graph

# Ensure dependencies are installed
os.system("pip install chromadb torch transformers pandas rdflib")

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="biomedical_faq")

# Load Hugging Face BioBERT Model (CPU-Only)
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to("cpu")  # Force CPU usage

# Load Knowledge Base (SPARQL)
kb = Graph()
kb.parse("biomedical_knowledge.ttl", format="turtle")

def query_sparql(query_text):
    query = f"""
    SELECT ?answer WHERE {{
        ?s ?p ?answer .
        FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{query_text}")))
    }}
    """
    results = kb.query(query)
    for row in results:
        return str(row["answer"])
    return None

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

# Define a Query Router
def query_router(query_text):
    # First, check SPARQL Knowledge Base
    kb_answer = query_sparql(query_text)
    if kb_answer:
        return kb_answer
    
    # If no KB match, check RAG
    rag_answer = query_rag(query_text)
    if rag_answer:
        return rag_answer
    
    # If no RAG match, fall back to MedBERT
    return query_medbert(query_text)

# Define a Query Function for RAG
def query_rag(query_text):
    results = collection.query(query_texts=[query_text], n_results=1)
    if results["documents"]:
        return results["documents"][0]
    return None

# Define a function to query MedBERT
def query_medbert(query_text):
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return "Generated response from MedBERT (Placeholder)"

# Run a Sample Query
if __name__ == "__main__":
    print("Testing query router with SPARQL KB, RAG, and MedBERT fallback:")
    user_query = "What is COVID-19?"
    response = query_router(user_query)
    print(f"User Query: {user_query}\nResponse: {response}")
