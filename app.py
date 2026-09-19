import os
import gc
import shutil
import uuid
import time
from datetime import datetime
import fitz  # PyMuPDF for fast PDF page rendering
import chromadb
import pandas as pd
import streamlit as st

import analytics

# 1. Page Configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Google Gemini Design System & CSS Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* Main application layout & Gemini radial background */
.stApp {
    background-color: #f8fafd;
    background-image: radial-gradient(ellipse 90% 65% at 50% 90%, #dbeafe 0%, rgba(248, 250, 253, 0) 70%);
    background-attachment: fixed;
    font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1f1f1f;
}

/* Hide Streamlit default header decoration & footer */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
footer {
    visibility: hidden !important;
    height: 0px !important;
}

/* Sidebar styling - soft #f0f4f9 background */
[data-testid="stSidebar"] {
    background-color: #f0f4f9 !important;
    border-right: 1px solid #e3e8ef !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
    padding-bottom: 1.5rem;
}

/* Gemini Sidebar Navigation Headers & Items */
.gemini-sidebar-title {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 6px 16px 6px;
}
.gemini-section-header {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #5f6368;
    margin-top: 1.25rem;
    margin-bottom: 0.45rem;
    padding-left: 6px;
}
.gemini-notebook-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
    border-radius: 10px;
    font-size: 0.88rem;
    color: #3c4043;
    background: transparent;
    transition: background 0.15s ease;
}
.gemini-notebook-item:hover {
    background: #e3e8ef;
}

/* User profile footer */
.gemini-user-profile {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 10px;
    margin-top: 2rem;
    border-top: 1px solid #e1e5ea;
}
.gemini-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1a73e8, #8ab4f8, #f28b82);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: white;
    font-size: 0.82rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
}
.gemini-user-name {
    font-size: 0.92rem;
    font-weight: 500;
    color: #1f1f1f;
}

/* Floating Gemini Pill Chat Input */
[data-testid="stChatInput"] {
    border-radius: 32px !important;
    background-color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06), 0 0 1px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #dcdfe4 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7facfa !important;
    box-shadow: 0 4px 18px rgba(66, 133, 244, 0.16) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
}

/* Chat bubble styling */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.85rem 0.25rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #e9eef6 !important;
    border-radius: 24px !important;
    padding: 0.85rem 1.35rem !important;
    margin-left: auto !important;
    max-width: 82% !important;
}

/* New chat pill button */
button[key="new_chat_btn"] {
    background-color: #e0e7f1 !important;
    border-radius: 24px !important;
    font-weight: 500 !important;
    color: #1f1f1f !important;
    padding: 9px 16px !important;
    border: none !important;
    transition: background 0.15s ease !important;
}
button[key="new_chat_btn"]:hover {
    background-color: #d2dbe6 !important;
}

/* Top upgrade pill */
button[key="top_creator_btn"],
button[key="top_back_btn"] {
    background-color: #c2e7ff !important;
    color: #001d35 !important;
    border-radius: 24px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    padding: 8px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: all 0.15s ease !important;
}
button[key="top_creator_btn"]:hover,
button[key="top_back_btn"]:hover {
    background-color: #aee0fd !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
}

/* Hero Suggestion Chips */
div:has(> button[key^="chip"]) button {
    background-color: #ffffff !important;
    border: 1px solid #e3e8ef !important;
    border-radius: 18px !important;
    padding: 16px 18px !important;
    text-align: left !important;
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    transition: all 0.2s ease !important;
    height: 100% !important;
}
div:has(> button[key^="chip"]) button:hover {
    border-color: #a8c7fa !important;
    background-color: #f8faff !important;
    box-shadow: 0 4px 14px rgba(66, 133, 244, 0.12) !important;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# 3. Environment & Settings Setup
GROQ_API_KEY = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = str(GROQ_API_KEY).strip().strip('"').strip("'")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is not set. Please add it in your Streamlit Cloud App Settings → Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.groq import Groq

BOOKS_DIR = os.environ.get("BOOKS_DIR", "./books")
CHROMA_DIR = "./chroma_db"


def safe_basename(filepath: str) -> str:
    """Cross-platform basename that handles both Windows (\\) and Unix (/) path separators.
    On Linux, os.path.basename('D:\\books\\file.pdf') returns 'D:\\books\\file.pdf' instead of 'file.pdf'.
    This function normalizes the separators first."""
    if not filepath:
        return filepath
    return filepath.replace('\\', '/').split('/')[-1]

ADMIN_PASSCODE = "medai2026"
try:
    if "ADMIN_PASSCODE" in st.secrets:
        ADMIN_PASSCODE = str(st.secrets["ADMIN_PASSCODE"]).strip().strip('"').strip("'")
except Exception:
    pass
if not ADMIN_PASSCODE:
    ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "medai2026")

@st.cache_data(ttl=3600)
def get_best_available_groq_model(api_key: str) -> str:
    """Queries Groq's live models endpoint with the API key and selects the best active generative chat model."""
    try:
        from groq import Groq as RawGroqClient
        client = RawGroqClient(api_key=api_key)
        all_models = [m.id for m in client.models.list().data if getattr(m, 'active', True)]
        print(f"Available Groq models: {all_models}", flush=True)

        # Exclude guard, moderation, classification, speech, audio, vision-only, embedding models
        non_chat_keywords = ["guard", "classif", "whisper", "embed", "safeguard", "moderation", "rerank"]
        valid_chat_models = [
            m for m in all_models
            if not any(kw in m.lower() for kw in non_chat_keywords)
        ]
        
        preferred_order = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        for candidate in preferred_order:
            if candidate in valid_chat_models:
                return candidate
        
        if valid_chat_models:
            return valid_chat_models[0]
    except Exception as e:
        print(f"Notice: Groq auto-detect model: {e}", flush=True)
    return "llama-3.1-8b-instant"

GROQ_MODEL = None
try:
    if "GROQ_MODEL" in st.secrets:
        GROQ_MODEL = str(st.secrets["GROQ_MODEL"]).strip().strip('"').strip("'")
except Exception:
    pass

if not GROQ_MODEL:
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "")

if not GROQ_MODEL:
    GROQ_MODEL = get_best_available_groq_model(GROQ_API_KEY)

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def get_client_ip() -> str:
    """Extracts client IP address using Streamlit 1.60 context or proxy headers."""
    try:
        if hasattr(st.context, "ip_address") and st.context.ip_address:
            return st.context.ip_address
    except Exception:
        pass

    try:
        if hasattr(st.context, "headers") and st.context.headers:
            for header_name in ["x-forwarded-for", "x-real-ip", "remote-addr"]:
                val = st.context.headers.get(header_name)
                if val:
                    return val.split(",")[0].strip()
    except Exception:
        pass

    return "127.0.0.1"


def get_client_user_agent() -> str:
    """Extracts device/browser user agent string."""
    try:
        if hasattr(st.context, "headers") and st.context.headers:
            return st.context.headers.get("user-agent", "Unknown Browser")
    except Exception:
        pass
    return "Unknown Browser"


def get_friendly_book_name(filename: str) -> str:
    """Returns a clean, student-friendly title with subject emoji for each textbook."""
    lower = filename.lower()
    if "api" in lower:
        return "🩺 API - Textbook of Medicine (9th Ed)"
    elif "gynecology" in lower:
        return "🤰 DC Dutta - Gynecology (6th Ed)"
    elif "obstetrics" in lower:
        return "👶 DC Dutta - Obstetrics"
    elif "harrison" in lower:
        return "🩺 Harrison's Internal Medicine (21st Ed)"
    elif "tripathy" in lower or "tripathi" in lower:
        return "💊 KD Tripathi - Pharmacology"
    elif "parks" in lower:
        return "🌍 Park's Preventive & Social Medicine (23rd Ed)"
    elif "sembulingum" in lower or "sembulingam" in lower:
        return "⚡ Sembulingam - Physiology"
    elif "srb" in lower:
        return "🔪 SRB's Manual of Surgery"
    elif "vishram" in lower and "vol 1" in lower:
        return "🏛️ Vishram Singh - Anatomy Vol 1 (Thorax & Upper Limb)"
    elif "vishram" in lower and "vol 2" in lower:
        return "🏛️ Vishram Singh - Anatomy Vol 2 (Abdomen & Lower Limb)"
    elif "vishram" in lower and "vol 3" in lower:
        return "🏛️ Vishram Singh - Anatomy Vol 3 (Head, Neck & Brain)"
    return filename


@st.cache_data(show_spinner=False, max_entries=120)
def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 140) -> bytes:
    """Renders a 1-indexed page from a PDF file as high-res PNG bytes."""
    if not os.path.exists(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        page_idx = max(0, min(page_num - 1, len(doc) - 1))
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception:
        return None


@st.cache_resource
def initialize_system():
    # Embedding model for local vector similarity
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
    
    # LLM for comprehensive, high-depth medical reasoning & text generation
    Settings.llm = Groq(
        model=GROQ_MODEL, 
        api_key=GROQ_API_KEY, 
        temperature=0.1
    )

    # Automatically download pre-built ChromaDB and textbooks from Hugging Face Dataset if not present
    if not os.path.exists(CHROMA_DIR) or not os.path.exists(BOOKS_DIR) or len(os.listdir(BOOKS_DIR)) == 0:
        from huggingface_hub import snapshot_download
        print("📥 Downloading pre-built ChromaDB and textbooks from Hugging Face dataset...", flush=True)
        snapshot_download(
            repo_id="Lurkingoddball/medical-ai-data",
            repo_type="dataset",
            local_dir=".",
            allow_patterns=["chroma_db/**", "books/**"]
        )

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    
    processed_collection = db.get_or_create_collection("processed_files")
    processed_files = set([item["file"] for item in processed_collection.get()["metadatas"] if "file" in item])

    chroma_collection = db.get_or_create_collection("medical_textbooks_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, stores_text=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = PyMuPDFReader()
    if os.path.exists(BOOKS_DIR):
        pdf_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".pdf")]
    else:
        pdf_files = []

    for filename in pdf_files:
        if filename in processed_files:
            continue

        file_path = os.path.join(BOOKS_DIR, filename)
        docs = reader.load_data(file_path)
        
        VectorStoreIndex.from_documents(
            docs, 
            storage_context=storage_context,
            show_progress=False
        )
        
        processed_collection.add(
            ids=[filename],
            documents=[filename],
            metadatas=[{"file": filename}]
        )
        processed_files.add(filename)
        del docs
        gc.collect()

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index, sorted(list(processed_files))


def prepare_search_query(user_raw_query: str, llm, conversation_history: list = None) -> str:
    """
    Translates, expands, and disambiguates user queries (English, Hinglish, Hindi, Marathi),
    corrects spelling mistakes/typos, and resolves follow-up references ('it', 'its treatment', 'management for iy')
    using conversation history into precise English medical textbook search keywords.
    """
    history_context = ""
    if conversation_history:
        recent_turns = []
        for msg in conversation_history[-4:]:
            role = "Student" if msg["role"] == "user" else "Assistant"
            content_snippet = msg["content"][:250].replace("\n", " ").strip()
            recent_turns.append(f"{role}: {content_snippet}")
        if recent_turns:
            history_context = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(recent_turns) + "\n\n"

    prompt = (
        "You are an expert clinical search query optimizer for medical textbooks.\n\n"
        f"{history_context}"
        f"LATEST STUDENT QUESTION (may contain spelling mistakes, typos, or follow-up pronouns like 'it', 'this', 'iy'):\n"
        f"'{user_raw_query}'\n\n"
        "OPTIMIZATION RULES:\n"
        "1. Context & Pronoun Resolution: If the student asks a follow-up question using pronouns or implied references (e.g., 'tell the management for iy', 'treatment of it', 'what are its causes', 'complications of that'), identify the specific medical disease/condition discussed in the previous conversation (e.g., Raynaud's phenomenon) and include it explicitly in the keywords.\n"
        "2. Typo & Spelling Correction: Automatically fix any spelling mistakes or typos (e.g., 'iy' -> 'it', 'pancreatitus' -> 'pancreatitis', 'diabtes' -> 'diabetes', 'otitis media treatement' -> 'otitis media treatment').\n"
        "3. Output Format: Produce 4 to 10 high-yield English medical keywords for vector textbook search (e.g., disease name, pathology, clinical signs, diagnosis, management protocols, drug classes).\n"
        "4. Output ONLY the search keywords separated by single spaces. Do not output markdown, punctuation, quotes, or conversational explanations."
    )
    try:
        res = llm.complete(prompt)
        terms = str(res).strip().replace('"', '').replace("'", "").replace('\n', ' ')
        return terms if terms else user_raw_query
    except Exception:
        return user_raw_query


def display_sources_and_page_viewer(sources_list, message_idx: int):
    """Renders cited sources and an interactive textbook page viewer with original diagrams."""
    if not sources_list:
        return

    with st.expander("📖 View Referenced Textbook Pages & Diagrams", expanded=False):
        valid_sources = []
        for s in sources_list:
            try:
                p = int(s.get("page", -1))
                if p > 0:
                    valid_sources.append(s)
            except (ValueError, TypeError):
                continue

        if valid_sources:
            st.markdown("#### 🔍 Original Textbook Pages")
            st.caption("Inspect authentic textbook pages with original diagrams, clinical tables, and anatomy figures:")
            
            options = [f"{s['book']} — Page {s['page']}" for s in valid_sources]
            selected_idx = st.selectbox(
                "Select page to inspect:",
                range(len(options)),
                format_func=lambda i: options[i],
                key=f"select_page_{message_idx}"
            )
            
            chosen_src = valid_sources[selected_idx]
            raw_filename = chosen_src.get("raw_file", "")
            pdf_path = os.path.join(BOOKS_DIR, raw_filename)
            page_num = int(chosen_src["page"])
            
            with st.spinner(f"Rendering page {page_num} from {chosen_src['book']}..."):
                img_bytes = render_pdf_page(pdf_path, page_num)
                
            if img_bytes:
                st.image(
                    img_bytes, 
                    caption=f"{chosen_src['book']} (Page {page_num})", 
                    use_container_width=True
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.download_button(
                        label=f"💾 Download Page {page_num}",
                        data=img_bytes,
                        file_name=f"{raw_filename}_page_{page_num}.png",
                        mime="image/png",
                        key=f"dl_btn_{message_idx}_{selected_idx}"
                    )
            else:
                st.info(f"Page preview unavailable for {raw_filename} (Page {page_num}).")

        with st.expander("📝 View Raw Text Excerpts & Citations", expanded=False):
            for src in sources_list:
                st.markdown(f"- **{src['book']}** | **Page:** `{src['page']}`")
                if src.get('snippet'):
                    st.caption(f"Snippet: {src['snippet']}...")


# Initialize system and database
with st.spinner("Initializing system and loading database..."):
    index, loaded_books = initialize_system()

# Ensure LLM uses the fresh GROQ_API_KEY from secrets on every rerun
Settings.llm = Groq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1
)

# Map friendly titles to actual filenames
book_options = {get_friendly_book_name(b): b for b in loaded_books}

# ==========================================
# GEMINI UNIFIED SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # 1. Brand Logo Header
    st.markdown("""
        <div class="gemini-sidebar-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C12 7.52285 7.52285 12 2 12C7.52285 12 12 16.4771 12 22C12 16.4771 16.4771 12 22 12C16.4771 12 12 7.52285 12 2Z" fill="url(#gemini-grad-side)"/>
                <defs>
                    <linearGradient id="gemini-grad-side" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#4285F4"/>
                        <stop offset="0.38" stop-color="#9B72CB"/>
                        <stop offset="0.78" stop-color="#D96570"/>
                        <stop offset="1" stop-color="#F2A600"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    """, unsafe_allow_html=True)

    # 2. "+ New chat" Pill Button
    if st.button("➕  New chat", key="new_chat_btn", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.session_state.current_view = "chat"
        st.rerun()

    # 3. Quick Utility Navigation
    with st.expander("🔍 Search & Medical Scope", expanded=False):
        st.caption("Search across textbooks and query history.")
        search_filter = st.text_input("Filter books:", placeholder="e.g. Medicine, Surgery...")

    # 4. Notebooks Section (Medical Subjects)
    st.markdown('<div class="gemini-section-header">Notebooks</div>', unsafe_allow_html=True)

    select_all = st.checkbox(
        f"Select All Textbooks ({len(loaded_books)} books)", 
        value=True,
        help="Search across all medical subjects simultaneously with balanced source diversity."
    )
    
    if select_all:
        selected_filenames = list(loaded_books)
        st.caption("✨ *All 11 medical notebooks active*")
    else:
        selected_display_names = st.multiselect(
            "Active Notebook(s):",
            options=list(book_options.keys()),
            default=list(book_options.keys())[:3] if len(book_options) >= 3 else list(book_options.keys()),
            help="Choose any 1 or more textbooks."
        )
        selected_filenames = [book_options[name] for name in selected_display_names]
        if not selected_filenames:
            st.error("⚠️ Please select at least one textbook!")

    # Display clean notebook list items like screenshot
    active_subjects = [get_friendly_book_name(b).split(" - ")[0].strip() for b in selected_filenames[:4]]
    for subj in active_subjects:
        st.markdown(f'<div class="gemini-notebook-item">📓 {subj}</div>', unsafe_allow_html=True)
    if len(selected_filenames) > 4:
        st.caption(f"+ {len(selected_filenames) - 4} more notebooks active")

    # 5. Recent Queries Section (pulled from SQLite logs)
    st.markdown('<div class="gemini-section-header">Recent</div>', unsafe_allow_html=True)
    recent_logs = []
    try:
        df_recent = analytics.get_all_logs()
        if not df_recent.empty:
            seen_q = set()
            unique_q = []
            for q in df_recent["user_query"].dropna().tolist():
                clean_q = q.strip()
                if clean_q and clean_q not in seen_q:
                    seen_q.add(clean_q)
                    unique_q.append(clean_q)
                    if len(unique_q) >= 6:
                        break
            recent_logs = unique_q
    except Exception:
        recent_logs = []

    if recent_logs:
        for idx, r_q in enumerate(recent_logs):
            truncated = r_q[:28] + "..." if len(r_q) > 28 else r_q
            if st.button(f"💬 {truncated}", key=f"rec_q_{idx}", use_container_width=True):
                st.session_state.pending_query = r_q
                st.session_state.current_view = "chat"
                st.rerun()
    else:
        st.caption("No recent queries yet.")

    # 6. Library Books Status expander
    with st.expander("📖 Library Books Status", expanded=False):
        for book in loaded_books:
            friendly = get_friendly_book_name(book)
            if "parks" in book.lower():
                st.caption(f"⚠️ **{friendly}**\n*(Scanned image PDF)*")
            else:
                st.caption(f"✅ **{friendly}**")

    # 7. User Profile Card at bottom (Aryan Jadhav + ⚙️)
    st.markdown("""
        <div class="gemini-user-profile">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="gemini-avatar">AJ</div>
                <div class="gemini-user-name">Aryan Jadhav</div>
            </div>
            <div style="font-size: 1.1rem; color: #5f6368;" title="Settings">⚙️</div>
        </div>
    """, unsafe_allow_html=True)


# Top Navigation Bar (Creator Portal Button)
top_col1, top_col2 = st.columns([5, 1])
with top_col1:
    st.empty()

with top_col2:
    if st.session_state.current_view == "chat":
        if st.button("✨ Creator Portal", key="top_creator_btn", help="Open Creator Analytics & Passcode Dashboard"):
            st.session_state.current_view = "creator"
            st.rerun()
    else:
        if st.button("← Back to Chat", key="top_back_btn"):
            st.session_state.current_view = "chat"
            st.rerun()


# ==========================================
# VIEW 1: CREATOR & ADMIN DASHBOARD
# ==========================================
if st.session_state.current_view == "creator":
    st.title("🔐 Creator & Admin Dashboard")
    st.caption("Private analytics, student queries, IP address tracking, and textbook metrics.")

    # Passcode Gate
    if not st.session_state.admin_authenticated:
        st.markdown("#### 🔒 Protected Creator Portal")
        st.info("This section is restricted to the application creator.")
        
        entered_pin = st.text_input(
            "Enter Admin Passcode:", 
            type="password", 
            help="Default passcode is 'medai2026'"
        )
        if st.button("Unlock Dashboard 🚀"):
            if entered_pin == ADMIN_PASSCODE:
                st.session_state.admin_authenticated = True
                st.success("Passcode verified! Welcome, Creator.")
                st.rerun()
            else:
                st.error("Incorrect passcode. Access denied.")
        st.stop()

    # Logged-in Admin Controls
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.success("✅ Logged in as Application Creator")
    with col_b:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    with col_c:
        if st.button("🔒 Lock Dashboard"):
            st.session_state.admin_authenticated = False
            st.rerun()

    st.markdown("---")

    # Fetch Metrics and Logs
    metrics = analytics.get_analytics_metrics()
    logs_df = analytics.get_all_logs()

    # KPI Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="👥 Unique Student IPs", value=metrics["unique_ips"])
    with m2:
        st.metric(label="💬 Total Questions Asked", value=metrics["total_queries"])
    with m3:
        st.metric(label="📚 Top Consulted Book", value=metrics["top_book"])
    with m4:
        st.metric(label="⚡ Avg Response Speed", value=f"{metrics['avg_latency']}s")

    st.markdown("---")

    # Charts Section
    if not logs_df.empty:
        ch1, ch2 = st.columns(2)
        
        with ch1:
            st.subheader("📊 Questions by Textbook Consulted")
            if metrics["book_counts"]:
                chart_data = pd.DataFrame(
                    list(metrics["book_counts"].items()),
                    columns=["Textbook", "Citations"]
                ).sort_values(by="Citations", ascending=False)
                st.bar_chart(chart_data.set_index("Textbook"))
            else:
                st.info("No citations recorded yet.")

        with ch2:
            st.subheader("📈 Student Activity Timeline")
            # Group by date
            logs_df["date"] = pd.to_datetime(logs_df["timestamp"]).dt.date
            activity_df = logs_df.groupby("date").size().reset_index(name="Queries")
            st.line_chart(activity_df.set_index("date"))

        st.markdown("---")

        # Student Query Audit Log Table
        st.subheader("📋 Student Interaction & IP Audit Log")
        
        # Search & Filter
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            search_query_term = st.text_input("🔍 Filter logs by Question, Keyword, or IP:")
        with filter_col2:
            st.write("")
            st.write("")
            csv_data = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Export Logs as CSV",
                data=csv_data,
                file_name=f"medical_ai_query_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        filtered_df = logs_df.copy()
        if search_query_term:
            term = search_query_term.lower()
            filtered_df = filtered_df[
                filtered_df["user_query"].str.lower().str.contains(term, na=False) |
                filtered_df["ip_address"].str.lower().str.contains(term, na=False) |
                filtered_df["search_keywords"].str.lower().str.contains(term, na=False)
            ]

        # Display Clean Audit Table
        display_cols = ["id", "timestamp", "ip_address", "user_query", "selected_scope", "latency_seconds"]
        st.dataframe(
            filtered_df[display_cols].rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "ip_address": "Client IP",
                "user_query": "Student Question",
                "selected_scope": "Selected Books",
                "latency_seconds": "Latency (s)"
            }),
            use_container_width=True,
            height=300
        )

        # Deep Query Inspector
        with st.expander("🔎 Deep Query Inspector (View full question, sources, and keywords)"):
            if not filtered_df.empty:
                selected_log_id = st.selectbox(
                    "Choose Log Entry ID to inspect:",
                    filtered_df["id"].tolist(),
                    format_func=lambda x: f"Log #{x} — {filtered_df[filtered_df['id'] == x]['user_query'].values[0][:60]}..."
                )
                entry = filtered_df[filtered_df["id"] == selected_log_id].iloc[0]
                
                i_col1, i_col2 = st.columns(2)
                with i_col1:
                    st.markdown(f"**IP Address:** `{entry['ip_address']}`")
                    st.markdown(f"**Timestamp:** `{entry['timestamp']}`")
                    st.markdown(f"**Session ID:** `{entry['session_id']}`")
                    st.markdown(f"**User Agent:** `{entry['user_agent']}`")
                with i_col2:
                    st.markdown(f"**Search Keywords Generated:** `{entry['search_keywords']}`")
                    st.markdown(f"**Scope Selected:** `{entry['selected_scope']}`")
                    st.markdown(f"**Response Length:** `{entry['response_length']} characters`")
                    st.markdown(f"**Latency:** `{entry['latency_seconds']} seconds`")
                
                st.markdown(f"**Full Question Asked:**\n> {entry['user_query']}")
                st.markdown(f"**Sources & Pages Cited:**\n```json\n{entry['sources_cited']}\n```")

    else:
        st.info("No queries have been logged yet. Ask questions in the Student Assistant view to see live analytics here!")

    st.markdown("---")
    with st.expander("⚙️ Database & Maintenance (Admin Only)"):
        st.warning("⚠️ **Danger Zone**: Only use these controls if you need to wipe cached data or re-index your PDF collection.")
        if st.button("🔄 Reset & Re-index Vector Database"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.cache_resource.clear()
            st.success("Vector database cleared! Reloading...")
            st.rerun()

        st.write("")
        if st.button("🗑️ Clear All Activity Logs"):
            analytics.clear_logs()
            st.success("All activity logs cleared successfully.")
            st.rerun()

    st.stop()


# ==========================================
# VIEW 2: GEMINI ASSISTANT INTERFACE
# ==========================================

# Active Scope Caption
if select_all:
    st.caption("✨ **Active Scope:** All 11 Medical Textbooks (Balanced Multi-Book Search)")
else:
    selected_names_str = ", ".join([get_friendly_book_name(f).split(" - ")[0] for f in selected_filenames])
    st.caption(f"🎯 **Active Scope:** {len(selected_filenames)} Selected Book(s): *{selected_names_str}*")

# Chat Interface State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hero State: When no messages have been sent yet
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style="text-align: center; margin-top: 14vh; margin-bottom: 2.5rem;">
            <h1 style="font-size: 2.85rem; font-weight: 500; letter-spacing: -0.025em; color: #1f1f1f; margin-bottom: 0.5rem;">
                Hello, How can i help you
            </h1>
            <p style="font-size: 1.08rem; color: #5f6368; max-width: 600px; margin: 0 auto;">
                Ask in-depth questions
            </p>
        </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg_idx, message in enumerate(st.session_state.messages):
    avatar_icon = "✨" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            display_sources_and_page_viewer(message["sources"], msg_idx)

# User Input Handling
input_text = st.chat_input("Ask anything...")
user_query = None

if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif input_text:
    user_query = input_text

if user_query:
    if not selected_filenames:
        st.warning("Please select at least one textbook in the sidebar first!")
        st.stop()

    client_ip = get_client_ip()
    user_agent = get_client_user_agent()
    query_start_time = time.time()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="✨"):
        scope_text = "all textbooks" if select_all else f"{len(selected_filenames)} selected book(s)"
        with st.spinner(f"Searching {scope_text}..."):
            
            # 1. Translate / expand query with conversation history context and typo correction
            search_keywords = prepare_search_query(
                user_raw_query=user_query, 
                llm=Settings.llm, 
                conversation_history=st.session_state.messages[:-1]
            )

            # 2. Retrieve candidates (top_k=16 allows thorough selection)
            retriever = index.as_retriever(similarity_top_k=16)
            retrieved_nodes = retriever.retrieve(search_keywords)

            # 3. Apply student's textbook selection filter & anti-monopoly diversity
            unique_nodes = []
            seen_texts = set()
            book_counts = {}

            for node in retrieved_nodes:
                raw_file = node.metadata.get('file_path', node.metadata.get('file_name', ''))
                book_filename = safe_basename(raw_file)

                # Filter strictly to the books selected by the student
                if book_filename not in selected_filenames:
                    continue

                # Deduplicate identical chunks
                norm_key = node.text[:80].strip()
                if norm_key in seen_texts:
                    continue

                # Limit chunks per textbook to maintain balance and prevent token explosion
                if select_all:
                    cnt = book_counts.get(book_filename, 0)
                    if cnt >= 2:
                        continue
                    book_counts[book_filename] = cnt + 1

                seen_texts.add(norm_key)
                unique_nodes.append(node)

                # Cap at 4 high-yield chunks to strictly respect Groq 8000 TPM limits
                if len(unique_nodes) >= 4:
                    break

            # 4. Format context with explicit textbook and page numbers
            context_blocks = []
            sources_list = []
            for node in unique_nodes:
                meta = node.metadata
                raw_file = meta.get('file_path', meta.get('file_name', 'Unknown Textbook'))
                raw_filename = safe_basename(raw_file) if raw_file else 'Unknown Textbook'
                friendly_book = get_friendly_book_name(raw_filename)
                page_num = meta.get('source') or meta.get('page_label') or meta.get('page') or 'N/A'
                snippet = node.text.strip()[:150].replace('\n', ' ')

                # Cap individual excerpt length to avoid token limit breach
                chunk_text = node.text.strip()
                if len(chunk_text) > 1200:
                    chunk_text = chunk_text[:1200] + "..."

                if len(chunk_text) > 0:
                    context_blocks.append(f"--- [Document: {friendly_book} | Page: {page_num}] ---\n{chunk_text}")
                
                src_entry = {
                    "book": friendly_book, 
                    "raw_file": raw_filename, 
                    "page": page_num, 
                    "snippet": snippet
                }
                if src_entry not in sources_list:
                    sources_list.append(src_entry)

            context_str = "\n\n".join(context_blocks)

            # 5. Format conversation history context (last 2 turns, capped for token safety)
            history_blocks = []
            for msg in st.session_state.messages[-3:-1]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_blocks.append(f"{role_label}: {msg['content'][:250]}...")
            history_str = "\n".join(history_blocks) if history_blocks else "None"

            # 6. High-yield, token-optimized medical prompt
            selected_names_formatted = ", ".join([get_friendly_book_name(b) for b in selected_filenames])
            system_prompt = f"""You are an authoritative Medical Reference Assistant for medical students and clinicians.
Provide structured, comprehensive, and clinically accurate answers based strictly on the provided textbook CONTEXT.
Active Scope: {selected_names_formatted}

CORE GUIDELINES:
1. CLINICAL DEPTH: Provide detailed, rigorous explanations (definitions, classifications, anatomy, physiology, clinical features, diagnostic workup, and treatment/management with exact drug names/dosages if given). Format with bold headers, structured tables, and clear bullet points.
2. CITATIONS: Cite the textbook and page number for each major section or finding: [Book Title, p. X]. Include a 'References / Sources Consulted' list at the very end.
3. MULTILINGUAL: Respond in the exact language/dialect used by the user (English, Hinglish, Hindi, Marathi) while keeping exact English medical terms intact.
4. HONESTY: If a specific sub-aspect is not in the excerpts, note it clearly. If the context has zero relevance, respond: 'This topic is not covered in the available textbook excerpts.'"""

            full_prompt = (
                f"{system_prompt}\n\n"
                f"===================\nCONVERSATION HISTORY:\n===================\n{history_str}\n\n"
                f"===================\nTEXTBOOK CONTEXT EXCERPTS:\n===================\n{context_str}\n\n"
                f"===================\nUSER QUESTION:\n===================\n{user_query}\n\n"
                f"DETAILED MEDICAL RESPONSE:\n"
            )

            # Stream the generated response token by token with rate limit protection
            def generate_stream():
                try:
                    for chunk in Settings.llm.stream_complete(full_prompt):
                        yield chunk.delta
                except Exception as e:
                    err_msg = str(e)
                    if "413" in err_msg or "rate_limit" in err_msg.lower():
                        yield "⚠️ **Rate Limit Notice**: Groq API token limit reached. Please wait a few seconds and retry."
                    else:
                        yield f"⚠️ **Error generating response**: {err_msg}"

            answer_text = st.write_stream(generate_stream())
            latency = time.time() - query_start_time

            # Log interaction to SQLite analytics database
            scope_summary = "All 11 Textbooks" if select_all else f"{len(selected_filenames)} selected: {selected_names_formatted[:60]}"
            analytics.log_interaction(
                session_id=st.session_state.session_id,
                ip_address=client_ip,
                user_query=user_query,
                search_keywords=search_keywords,
                selected_scope=scope_summary,
                sources_cited=sources_list,
                response_length=len(answer_text),
                latency_seconds=latency,
                user_agent=user_agent
            )

            # Display interactive page viewer & debugging metadata box
            display_sources_and_page_viewer(sources_list, len(st.session_state.messages))

            # Save assistant message with sources to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer_text,
                "sources": sources_list
            })