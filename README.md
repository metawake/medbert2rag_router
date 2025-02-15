# Biomedical Query Router: SPARQL, RAG, and MedBERT

## Overview
This project demonstrates a **semantic search router** that efficiently organizes and prioritizes responses to natural language queries using three different data sources:
1. **Knowledge Base (SPARQL)**: Structured, curated data stored in an RDF knowledge graph.
2. **Retrieval-Augmented Generation (RAG) with ChromaDB**: Searches pre-indexed biomedical FAQ data when KB does not have an answer.
3. **MedBERT (LLM Fallback)**: Uses a **pre-trained BioBERT model** to generate answers when no relevant information is found in KB or RAG.

By leveraging these layers, the system ensures **high accuracy, context relevance, and explainability**, which are critical in biomedical research and healthcare.

---
## Why Semantic Search is Important
Traditional keyword-based search systems often **fail to capture meaning** and **struggle with synonyms, variations, and complex queries**. 

This project employs **semantic search** techniques to:
- Retrieve **structured data first** from a knowledge graph for highly reliable results.
- Use **vector embeddings** to find semantically similar content when KB lacks a direct match.
- Generate **natural language responses** using a biomedical-specific **LLM fallback**.

---
## Query Routing Strategy
### **1️⃣ SPARQL Knowledge Base (KB) Search**
- The first step in query resolution.
- Searches **structured RDF data** using SPARQL queries.
- Useful for well-defined biomedical concepts, e.g.,
  - "What are the symptoms of COVID-19?"
  - "What drugs treat influenza?"
- If the KB contains the answer, the system **returns it immediately**.

### **2️⃣ Retrieval-Augmented Generation (RAG) with ChromaDB**
- If the **KB does not contain the answer**, the query is **searched in a vector database**.
- RAG matches **similar questions** from a pre-indexed biomedical FAQ dataset.
- Example queries that can benefit from RAG:
  - "How does ibuprofen work?"
  - "What is the mechanism of action of Tamiflu?"
- The system retrieves the best-matching document and provides **context-aware responses**.

### **3️⃣ MedBERT LLM Fallback**
- If **both KB and RAG fail**, the system falls back to **MedBERT**, a biomedical language model.
- MedBERT generates answers dynamically based on its trained knowledge.
- This is useful for **complex or novel queries**:
  - "How does COVID-19 impact lung function over time?"
  - "Are there any recent studies on AI-driven drug discovery?"
- MedBERT responses are less structured but **can generate insights beyond stored data**.

---
## Project Structure
```
├── biomedical_faqs.csv        # FAQ data for RAG-based retrieval
├── biomedical_knowledge.ttl   # RDF knowledge base (SPARQL)
├── nlp_to_rag_v2.py           # Main script with query routing
├── requirements.txt           # Dependencies
├── README.md                  # This file
```

---
## How to Run Locally
### **1. Set Up a Python Virtual Environment**
```bash
python -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Query System**
```bash
python nlp_to_rag_v2.py
```

### **4. Example Queries**
Try running the following example queries:
```python
query_router("What is COVID-19?")
query_router("What are the symptoms of flu?")
query_router("How does ibuprofen work?")
```

---
## Future Enhancements
- **Expand the Knowledge Base** with more biomedical relations.
- **Enhance the RAG model** with PubMed abstracts.
- **Improve MedBERT Fine-Tuning** to generate better responses.

---
This project provides semantic search solution** for biomedical data retrieval. 🚀

