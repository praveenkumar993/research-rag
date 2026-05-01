from __future__ import annotations
import os
import time
import fitz
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

load_dotenv()

PAPERS_DIR      = Path("papers")
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE      = 1000  # characters per chunk (can adjust based on typical paper length)
CHUNK_OVERLAP   = 50   # characters of overlap between chunks (helps with context continuity)
BATCH_SIZE      = 100  # can go much higher now — no rate limits!

PAPER_NAME_MAP = {
    "attention_is_all_you_need": "Attention Is All You Need",
    "bert":                      "BERT",
    "gpt3":                      "GPT-3",
    "word2vec":                  "Word2Vec",
    "resnet":                    "ResNet",
    "lora":                      "LoRA",
    "rag":                       "RAG (Retrieval-Augmented Generation)",
    "chain_of_thought":          "Chain-of-Thought Prompting",
}


class HuggingFaceEmbeddings:
    """
    Local embeddings using sentence-transformers.
    Runs on your CPU/GPU — completely free, no API, no rate limits.
    Model: all-MiniLM-L6-v2
      - 384 dimensions
      - Very fast on CPU
      - Excellent for semantic similarity / RAG
      - Downloads once (~90MB), cached locally forever
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  🤗 Loading HuggingFace model: {model_name} (downloads once ~90MB)...")
        self.model = SentenceTransformer(model_name)
        print(f"  ✅ Model loaded!\n")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def get_paper_display_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if len(stem) > 3 and stem[:2].isdigit() and stem[2] == "_":
        stem = stem[3:]
    return PAPER_NAME_MAP.get(stem, stem.replace("_", " ").title())


def batch_embed_and_store(
    all_docs: list[Document],
    embeddings: HuggingFaceEmbeddings,
    vectorstore_dir: str,
    collection_name: str,
) -> Chroma:
    db = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    already_stored = db._collection.count()
    if already_stored > 0:
        print(f"  ⏭️  Found {already_stored} chunks already stored. Resuming...\n")

    remaining_docs = all_docs[already_stored:]

    if not remaining_docs:
        print("  ✅ All chunks already embedded. Nothing to do!")
        return db

    batches = [remaining_docs[i : i + BATCH_SIZE] for i in range(0, len(remaining_docs), BATCH_SIZE)]

    for idx, batch in enumerate(batches, start=1):
        texts     = [d.page_content for d in batch]
        metadatas = [d.metadata     for d in batch]
        print(f"  📡 Embedding batch {idx}/{len(batches)} ({len(texts)} chunks)...")
        db.add_texts(texts=texts, metadatas=metadatas)
        # No sleep needed — running locally!

    return db


def ingest_papers():
    print("🔍 Scanning papers directory...")
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDFs found in /papers directory.")
        return

    print(f"📄 Found {len(pdf_files)} papers. Starting ingestion...\n")

    embeddings = HuggingFaceEmbeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs: list[Document] = []

    for pdf_path in pdf_files:
        display_name = get_paper_display_name(pdf_path.name)
        print(f"  ⚙️  Processing: {display_name}")
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print(f"  ⚠️  Skipped (empty text): {pdf_path.name}")
            continue
        chunks = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"paper_name": display_name, "source": pdf_path.name}],
        )
        all_docs.extend(chunks)
        print(f"  ✅  {len(chunks)} chunks created for: {display_name}")

    print(f"\n📦 Total chunks to embed: {len(all_docs)}")
    print("🚀 Embedding locally with HuggingFace (no rate limits)...\n")

    batch_embed_and_store(
        all_docs=all_docs,
        embeddings=embeddings,
        vectorstore_dir=VECTORSTORE_DIR,
        collection_name="research_papers",
    )

    print(f"\n✅ Ingestion complete! {len(all_docs)} chunks stored in ChromaDB.")


if __name__ == "__main__":
    ingest_papers()