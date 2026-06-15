# RAG-Based Document Q&A System

An AI-powered document question answering system built with LangChain, Groq (Llama 3), ChromaDB, and Streamlit.

## Features
- Upload any PDF document
- Ask questions in plain English
- Get accurate answers with source references
- Multi-turn conversation with chat history memory
- 100% free stack (Groq + HuggingFace embeddings)

## Tech Stack
- **LangChain** — LLM orchestration
- **Groq (Llama 3.3-70b)** — Free LLM backend
- **HuggingFace Embeddings** — Local sentence embeddings
- **ChromaDB** — Vector database
- **Streamlit** — Web UI

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/abishxavier/rag-document-qa.git
cd rag-document-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

## How It Works
1. Upload a PDF document
2. System extracts and chunks the text
3. Chunks are embedded and stored in ChromaDB
4. Ask questions — relevant chunks are retrieved
5. Groq LLM answers using only those chunks