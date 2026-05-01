from __future__ import annotations
import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
TOP_K = 10

SYSTEM_PROMPT = """You are a senior AI research assistant specializing exclusively in the following research papers:
- Attention Is All You Need
- BERT
- GPT-3
- Word2Vec
- ResNet
- LoRA
- RAG (Retrieval-Augmented Generation)
- Chain-of-Thought Prompting

You ONLY answer questions related to these research papers and AI/ML concepts covered within them.

If a question is outside the scope of these papers (like general knowledge, politics, current events, people, places etc.), respond with:
"⚠️ I'm a Research Assistant trained exclusively on AI/ML research papers. Your question appears to be outside my scope. I can only answer questions related to: Attention Is All You Need, BERT, GPT-3, Word2Vec, ResNet, LoRA, RAG, Chain-of-Thought Prompting. Try asking something like 'How does BERT work?' or 'What is the attention mechanism?'"

When answering questions that ARE within scope:
- Give a thorough, detailed, and well-structured explanation
- Break your answer into clear sections or paragraphs where appropriate
- Explain key concepts, how they work, why they matter, and any important details from the paper
- Include specific details, numbers, formula descriptions, or examples mentioned in the paper if relevant
- Write at least 3-5 full paragraphs for any technical question
- Do NOT give one-liner or short answers — always explain in depth as if teaching someone

Always mention which paper(s) your answer is sourced from at the end of your response in this format:
📄 Source Paper(s): [Paper Name]

Context from papers:
{context}

Question: {question}

Answer:"""

def query_papers(question: str, chain) -> str:
    for attempt in range(3):
        try:
            return chain.invoke(question)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = (attempt + 1) * 30
                print(f"⏳ Rate limited. Waiting {wait}s before retry {attempt+2}/3...")
                time.sleep(wait)
            else:
                print(f"❌ Query error: {e}")
                raise e

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name="research_papers",
    )


def get_loaded_papers(vectorstore) -> list[str]:
    import chromadb
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    col = client.get_collection("research_papers")
    results = col.get(include=["metadatas"])
    names = set()
    for meta in results["metadatas"]:
        if meta and "paper_name" in meta:
            names.add(meta["paper_name"])
    return sorted(list(names))


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
    max_output_tokens=2048,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT,
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('paper_name', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def query_papers(question: str, chain) -> str:
    return chain.invoke(question)