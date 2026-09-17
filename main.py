import os
import gc
import chromadb

# 1. Read API key from environment variable (set via HF Spaces Secrets or local .env)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Add it in HF Spaces → Settings → Secrets.")

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.groq import Groq

BOOKS_DIR = os.environ.get("BOOKS_DIR", "./books")
CHROMA_DIR = "./chroma_db"

def initialize_system():
    print("⚡ Configuring local embeddings & Groq LLM...", flush=True)
    
    # Configure Embeddings & LLM globally
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
    Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    
    processed_collection = db.get_or_create_collection("processed_files")
    processed_files = set([item["file"] for item in processed_collection.get()["metadatas"] if "file" in item])

    chroma_collection = db.get_or_create_collection("medical_textbooks_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, stores_text=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = PyMuPDFReader()
    pdf_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".pdf")]
    total_books = len(pdf_files)

    print(f"\n📖 Found {total_books} total books. Already completed: {len(processed_files)}\n", flush=True)

    for idx, filename in enumerate(pdf_files, start=1):
        if filename in processed_files:
            print(f"⏩ [{idx}/{total_books}] Already indexed: {filename} (Skipping)", flush=True)
            continue

        file_path = os.path.join(BOOKS_DIR, filename)
        print(f"\n⏳ [{idx}/{total_books}] Processing: {filename}...", flush=True)
        
        docs = reader.load_data(file_path)
        
        VectorStoreIndex.from_documents(
            docs, 
            storage_context=storage_context,
            show_progress=True
        )
        
        processed_collection.add(
            ids=[filename],
            documents=[filename],
            metadatas=[{"file": filename}]
        )
        processed_files.add(filename)

        del docs
        gc.collect()
        print(f"✅ Finished [{idx}/{total_books}]: {filename}", flush=True)
        print("-" * 50, flush=True)

    print("\n🎉 ALL BOOKS LOADED! Medical system ready.", flush=True)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

def ask_question(index, query_text):
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        system_prompt=(
            "You are a strict medical AI textbook assistant.\n"
            "1. Answer strictly using ONLY the provided context.\n"
            "2. If the answer is not in the context, say: 'Information not found in the uploaded books.'\n"
            "3. Answer in the EXACT language/dialect of the question (English, Hinglish, Hindi, Marathi)."
        )
    )
    return query_engine.query(query_text)

if __name__ == "__main__":
    index = initialize_system()
    print("\nSystem Ready! Ask any question in English, Hinglish, Hindi, or Marathi.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("Ask Question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        print(f"\nAnswer:\n{ask_question(index, user_query)}\n" + "-"*50)