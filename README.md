# Research RAG — AI Paper Assistant

A RAG-based system to query 10 foundational AI research papers using Google Gemini + ChromaDB + LangChain.

---

## Project Structure

```
research_rag/
├── papers/                  ← Drop your 10 PDFs here
├── vectorstore/             ← ChromaDB auto-creates this
├── frontend/
│   └── index.html           ← Chat UI
├── src/
│   ├── ingest.py            ← PDF ingestion + embedding
│   ├── rag_engine.py        ← RAG query logic
│   └── main.py              ← FastAPI server
├── .env                     ← Your API key (never commit this)
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## Setup (Local — No Docker)

### 1. Clone and create virtual environment

```bash
git clone <your-repo>
cd research_rag
python -m venv venv
source venv/bin/activate     
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Get your key from: https://aistudio.google.com/app/apikey

### 4. Add your PDFs

Drop all 10 PDF files into the `papers/` folder using this naming convention:
```
papers/
├── 09_attention_is_all_you_need.pdf
├── 10_bert.pdf
├── 12_gpt3.pdf
├── 05_word2vec.pdf
├── 08_resnet.pdf
├── 22_lora.pdf
├── 26_rag.pdf
├── 21_chain_of_thought.pdf
```

### 5. Run ingestion (ONE TIME ONLY)

```bash
python src/ingest.py
```

You will see output like:
```
🔍 Scanning papers directory...
📄 Found 10 papers. Starting ingestion...
  ⚙️  Processing: Attention Is All You Need
  ✅  94 chunks created
  ...
✅ Ingestion complete! 847 chunks stored in ChromaDB.
```

> Only run this once. The vectorstore persists on disk in `vectorstore/`.
> If you add new papers later, delete the `vectorstore/` folder and re-run.

### 6. Start the server

```bash
uvicorn src.main:app --reload --port 8000
```

### 7. Open the UI

Visit: http://localhost:8000

---

## Setup (Docker)

### 1. Build and run with Docker Compose

```bash
cp .env.example .env
# Add your GEMINI_API_KEY to .env

docker-compose up --build
```

### 2. Run ingestion inside the container

```bash
docker-compose exec research-rag python src/ingest.py
```

### 3. Open the UI

Visit: http://localhost:8000

---

## Deployment on Render

### Prerequisites
- Push your code to a GitHub repo (do NOT commit `.env` or `papers/` or `vectorstore/`)
- Your vectorstore must be pre-built locally and committed, OR you run ingestion after deploy

### Steps

1. Go to https://render.com and create a new account
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render will auto-detect the `render.yaml`
5. In the dashboard, go to **Environment** and add:
   - `GEMINI_API_KEY` = your key
6. Click **Deploy**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check if RAG system is ready |
| GET | `/api/papers` | List all indexed paper names |
| POST | `/api/query` | Ask a question |

### POST /api/query

**Request:**
```json
{
  "question": "What is the attention mechanism?"
}
```

**Response:**
```json
{
  "question": "What is the attention mechanism?",
  "answer": "The attention mechanism... 📄 Source Paper(s): Attention Is All You Need"
}
```

---

## Adding More Papers

1. Add new PDFs to `papers/`
2. Update `PAPER_NAME_MAP` in `src/ingest.py` with the new filename → display name mapping
3. Delete the `vectorstore/` folder
4. Re-run `python src/ingest.py`
5. Restart the server

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Pro |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| PDF Parsing | PyMuPDF (fitz) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| Container | Docker |
| Deployment | Render |
