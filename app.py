import streamlit as st
import os
from rag_pipeline import process_pdf, build_vectorstore, build_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocuMind RAG | Emerald Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Green & White CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base Light Background with Subtle Emerald Ambient Glow */
.stApp {
    background-color: #F8FAFC;
    background-image: 
        radial-gradient(circle at 10% 12%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 90% 18%, rgba(52, 211, 153, 0.07) 0%, transparent 45%),
        radial-gradient(circle at 50% 95%, rgba(5, 150, 105, 0.05) 0%, transparent 50%);
    background-attachment: fixed;
    color: #0F172A;
}

/* Sidebar styling - Clean White with Emerald Accent */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid rgba(16, 185, 129, 0.18);
    box-shadow: 2px 0 12px rgba(16, 185, 129, 0.03);
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(16, 185, 129, 0.15);
    margin: 1.2rem 0;
}

/* Header & Gradient Typography */
.gradient-header {
    background: linear-gradient(135deg, #065F46 0%, #059669 45%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.sub-header {
    color: #475569;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* Badges & Pills */
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(16, 185, 129, 0.10);
    color: #065F46;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin-bottom: 0.8rem;
}

.pill-badge-green {
    background: rgba(16, 185, 129, 0.16);
    color: #047857;
    border: 1px solid #10B981;
}

.pill-badge-amber {
    background: rgba(245, 158, 11, 0.12);
    color: #B45309;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Crisp White Cards with Emerald Border Hints */
.glass-card {
    background: #FFFFFF;
    border: 1px solid rgba(16, 185, 129, 0.22);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px -2px rgba(16, 185, 129, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    border-color: #10B981;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px -2px rgba(16, 185, 129, 0.15);
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.feature-box {
    background: #F0FDF4;
    border: 1px solid rgba(16, 185, 129, 0.22);
    border-radius: 12px;
    padding: 1.2rem;
    transition: all 0.2s ease;
}

.feature-box:hover {
    background: #DCFCE7;
    border-color: #10B981;
    transform: translateY(-2px);
}

.feature-title {
    font-size: 1rem;
    font-weight: 700;
    color: #065F46;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.feature-desc {
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.5;
}

/* File uploader modern style */
[data-testid="stFileUploader"] {
    background: #F8FAFC;
    border-radius: 12px;
    border: 1.5px dashed rgba(16, 185, 129, 0.5);
    padding: 0.8rem;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #059669;
    background: #F0FDF4;
}

/* Vibrant Emerald Gradient Buttons */
.stButton > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: #FFFFFF !important;
    font-weight: 600;
    border: none;
    border-radius: 9px;
    padding: 0.6rem 0.5rem;
    box-shadow: 0 4px 14px -2px rgba(16, 185, 129, 0.45);
    transition: all 0.2s ease;
    width: 100% !important;
    height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.88rem !important;
    white-space: nowrap !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    box-shadow: 0 6px 20px -2px rgba(16, 185, 129, 0.65);
    transform: translateY(-1px);
    color: #FFFFFF !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    border-radius: 14px;
    background: #FFFFFF !important;
    border: 1.5px solid rgba(16, 185, 129, 0.45) !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.22) !important;
}

/* Chat Messages Bubble Styling */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}

/* Source Citation Card */
.source-card {
    background: #F0FDF4;
    border-left: 3.5px solid #10B981;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #334155;
    border-top: 1px solid rgba(16, 185, 129, 0.15);
    border-right: 1px solid rgba(16, 185, 129, 0.15);
    border-bottom: 1px solid rgba(16, 185, 129, 0.15);
}

.source-header {
    font-weight: 700;
    color: #065F46;
    margin-bottom: 0.3rem;
    display: flex;
    justify-content: space-between;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #F8FAFC;
}
::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session state initialization
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None

# Sidebar Setup
with st.sidebar:
    st.markdown('<div class="gradient-header" style="font-size: 1.4rem;">🌿 DocuMind RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header" style="margin-bottom: 0.8rem;">Intelligent Multi-Turn PDF Assistant</div>', unsafe_allow_html=True)

    # Document Upload Section
    st.markdown("### 📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf", help="Max file size 200MB")

    col_btn, col_clear = st.columns(2)
    
    with col_btn:
        process_clicked = st.button("🚀 Process PDF", use_container_width=True)

    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    active_key = os.environ.get("GROQ_API_KEY", "").strip()

    if uploaded_file and process_clicked:
        if not active_key:
            st.error("⚠️ GROQ_API_KEY is not configured in your environment. Please add it to your .env file.")
        else:
            with st.spinner("🌿 Extracting, chunking & generating vector embeddings..."):
                try:
                    chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = build_vectorstore(chunks)
                    st.session_state.chain = build_qa_chain(st.session_state.vectorstore, api_key=active_key)
                    st.session_state.chat_history = []
                    st.session_state.doc_meta = {
                        "name": uploaded_file.name,
                        "chunks": len(chunks),
                        "size_kb": round(len(uploaded_file.getvalue()) / 1024, 1)
                    }
                    st.success(f"✓ Indexed {len(chunks)} chunks successfully!")
                except Exception as ex:
                    st.error(f"Error processing document: {str(ex)}")

    st.markdown("---")

    # Document Status and Model Architecture Card
    if st.session_state.doc_meta:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="pill-badge pill-badge-green">● Document Active</span>
                <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-top: 0.3rem;">{st.session_state.doc_meta['name']}</div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.2rem;">
                    Chunks: <b>{st.session_state.doc_meta['chunks']}</b> • Size: <b>{st.session_state.doc_meta['size_kb']} KB</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="glass-card">
                <span class="pill-badge pill-badge-amber">○ Awaiting Document</span>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">
                    Upload a PDF to vectorize and query with Groq LPUs.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Tech Stack Footnote
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5; margin-top: 1rem;">
            <b>Architecture</b>: ChromaDB • HuggingFace Embeddings (all-MiniLM-L6-v2) • Groq LPU Inference
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Page Layout
col_header, col_badges = st.columns([2.5, 1.5])
with col_header:
    st.markdown('<div class="gradient-header">🌿 DocuMind RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Conversational Question Answering with Semantic Vector Retrieval & Citation Grounding</div>', unsafe_allow_html=True)

with col_badges:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 0.5rem;">
            <span class="pill-badge">Groq LPUs</span>
            <span class="pill-badge">ChromaDB</span>
            <span class="pill-badge">LangChain</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chat or Empty State
if st.session_state.chain:
    # Display Chat History
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🌿"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

    # User Query Input
    if question := st.chat_input("Ask any question about your indexed document..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Synthesizing context & generating grounded answer..."):
                try:
                    # Dynamically ensure chain has current key if user updated it
                    if active_key and st.session_state.vectorstore:
                        st.session_state.chain = build_qa_chain(st.session_state.vectorstore, api_key=active_key)
                    
                    result = st.session_state.chain({
                        "input": question,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = result["answer"]
                    sources = result.get("context", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 Retrieved Context Sources ({len(sources)} Chunks)", expanded=False):
                            for i, doc in enumerate(sources):
                                page = doc.metadata.get('page', 'Unknown')
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span>📌 Reference #{i+1}</span>
                                            <span>Page {page}</span>
                                        </div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #334155;">
                                            {doc.page_content[:350]}...
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.session_state.chat_history.extend([
                        HumanMessage(content=question),
                        AIMessage(content=answer)
                    ])

                except Exception as err:
                    err_msg = str(err)
                    if "401" in err_msg or "invalid_api_key" in err_msg.lower() or "AuthenticationError" in err_msg:
                        st.error("🔑 **Authentication Error**: The Groq API key is invalid or expired. Please update your `GROQ_API_KEY` in the `.env` file or hosting environment variables.")
                        st.markdown("[Get a free Groq API Key at console.groq.com ↗](https://console.groq.com)")
                    elif "rate_limit" in err_msg.lower():
                        st.warning("⚠️ **Rate Limit Exceeded**: Groq request limit reached. Please wait a few seconds before trying again.")
                    else:
                        st.error(f"⚠️ Error executing query: {err_msg}")

else:
    # Rich Modern Hero Empty State in Crisp White & Emerald Green
    st.markdown(
        """
        <div class="glass-card" style="padding: 2.2rem; border-color: rgba(16, 185, 129, 0.35);">
            <div style="font-size: 1.35rem; font-weight: 700; color: #065F46; margin-bottom: 0.5rem;">
                🚀 Welcome to DocuMind RAG
            </div>
            <div style="font-size: 0.95rem; color: #475569; max-width: 720px; line-height: 1.6;">
                Transform any PDF document into an interactive, multi-turn conversational knowledge base.
                Powered by state-of-the-art dense semantic retrieval and ultra-fast inference with Groq Llama 3.3.
            </div>
            <div class="feature-grid">
                <div class="feature-box">
                    <div class="feature-title">🎯 Semantic Search</div>
                    <div class="feature-desc">HuggingFace sentence-transformers chunk and embed your document into high-dimensional vector space.</div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">⚡ Ultra-Fast Groq LPUs</div>
                    <div class="feature-desc">Sub-second grounded responses generated by Llama-3.3-70B with context window synthesis.</div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">🔍 Source Citations</div>
                    <div class="feature-desc">Inspect exact chunk excerpts and document page citations for every generated answer.</div>
                </div>
            </div>
        </div>
        <div style="text-align: center; color: #64748B; font-size: 0.88rem; margin-top: 1.5rem;">
            👈 <b>Get Started</b>: Open the sidebar, upload a PDF document, and click <b>Process PDF</b>.
        </div>
        """,
        unsafe_allow_html=True
    )