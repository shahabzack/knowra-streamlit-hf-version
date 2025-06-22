import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv
import io
from typing import List
from utils import load_and_split_pdf, create_vector_store, create_rag_chain, PageFilteredRetriever,get_page_count
from langchain.chains import create_retrieval_chain
import cachetools

# Page configuration
st.set_page_config(
    page_title="Knowra: Your aura of answers",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="📖"
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file. Please set it.")
    st.stop()

# --- CSS Styling (unchanged) ---
st.markdown("""
<style>
    .stApp { 
        background-color: #0e1117; 
        color: #fafafa; 
        font-family: 'Segoe UI', sans-serif;
    }
    .main .block-container {
        padding: 1rem;
        max-width: 800px;
    }
    .main-header { 
        font-size: 1.8rem; 
        color: #fafafa; 
        margin: 0 0 0.5rem 0; 
        font-weight: 600; 
        text-align: center;
    }
    .sub-header { 
        font-size: 1.2rem; 
        color: #fafafa; 
        margin: 1rem 0 0.5rem 0; 
        font-weight: 500;
    }
    .info-text { 
        font-size: 0.9rem; 
        color: #a0a0a0; 
        text-align: center; 
        margin: 0 0 1.5rem 0;
    }
    [data-testid="stSidebar"] { 
        background-color: #1e2329; 
        padding: 1rem 0.5rem;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #fafafa;
        font-size: 1.1rem;
        margin: 0 0 1rem 0;
    }
    .stButton > button { 
        background-color: #1A4D3A; 
        color: #ffffff; 
        border-radius: 8px; 
        padding: 0.5rem 1rem; 
        font-weight: 500; 
        border: none; 
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { 
        background-color: #00b894; 
        transform: translateY(-1px);
    }
    .stTextInput > div > div > input { 
        background-color: #262730; 
        color: #fafafa; 
        border-radius: 8px; 
        padding: 0.75rem 1rem; 
        border: 1px solid #3a3a3a;
        font-size: 0.9rem;
    }
    .chat-container { 
        background-color: #1e2329; 
        border-radius: 12px; 
        padding: 1rem; 
        margin: 1rem 0; 
        max-height: 400px; 
        overflow-y: auto; 
        border: 1px solid #2a2a2a;
        box-sizing: border-box;
    }
    .message-wrapper {
        display: flex;
        margin: 0.5rem 0;
    }
    .user-message-wrapper {
        justify-content: flex-end;
    }
    .assistant-message-wrapper {
        justify-content: flex-start;
    }
    .message-bubble {
        max-width: 75%;
        word-wrap: break-word;
        padding: 0.75rem 1rem;
        border-radius: 16px;
        box-sizing: border-box;
    }
    .user-message { 
        background-color: #1A4D3A; 
        color: #ffffff;
        border-bottom-right-radius: 4px;
    }
    .assistant-message { 
        background-color: #262730; 
        color: #fafafa;
        border-bottom-left-radius: 4px;
        margin-bottom: 10px;
    }
    .message-header {
        font-size: 0.75rem;
        color: #a0a0a0;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    .timestamp { 
        font-size: 0.7rem; 
        color: #a0a0a0; 
        margin-top: 0.25rem;
        opacity: 0.7;
    }
    .chat-input-container {
        background-color: #1e2329;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #2a2a2a;
    }
    .info-box { 
        background-color: #1e2329; 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid #2a2a2a; 
        margin: 1rem 0;
        color: #fafafa;
    }
    .welcome-container {
        background-color: #1e2329;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #2a2a2a;
        text-align: center;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        .main-header { 
            font-size: 1.5rem; 
        }
        .chat-container { 
            max-height: 300px; 
            padding: 0.75rem;
        }
        .message-bubble {
            max-width: 85%;
            padding: 0.6rem 0.8rem;
        }
        .chat-input-container {
            padding: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
    st.session_state.filename = None
    st.session_state.pdf_processed = False
    st.session_state.show_chat = False
    st.session_state.preview_mode = "Full Document"
    st.session_state.start_page = 0
    st.session_state.end_page = 0
    st.session_state.chat_history = []
    st.session_state.user_name = ""
    st.session_state.name_prompted = False
    st.session_state.vector_store = None
    st.session_state.document_chain = None
    st.session_state.total_pages = 0

# Initialize RAG chain if not already done
if st.session_state.document_chain is None:
    st.session_state.document_chain = create_rag_chain()

# --- App Header ---
st.markdown('<h1 class="main-header">📖 Knowra: Your aura of answers</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Unlock insights from your PDFs with intelligent chat</p>', unsafe_allow_html=True)

# --- Sidebar for Document Upload and Settings ---
with st.sidebar:
    st.markdown('<h2 class="sub-header">📄 Document Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        if st.session_state.filename != uploaded_file.name:
            st.session_state.pdf_bytes = uploaded_file.read()
            st.session_state.filename = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.show_chat = False
            st.session_state.chat_history = []
            st.session_state.name_prompted = False
            st.session_state.user_name = ""
            st.session_state.vector_store = None
            st.session_state.total_pages = get_page_count(st.session_state.pdf_bytes)

        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                documents = load_and_split_pdf(io.BytesIO(st.session_state.pdf_bytes), st.session_state.filename)
                if not documents:
                    st.error("Could not extract text from the PDF.")
                else:
                    st.session_state.vector_store = create_vector_store(documents)
                    if st.session_state.vector_store:
                        st.session_state.pdf_processed = True
                        st.success(f"✅ Loaded {st.session_state.total_pages} pages!")
                    else:
                        st.error("Failed to create vector store.")

        if st.session_state.pdf_processed:
            total_pages = st.session_state.total_pages
            st.markdown("**📊 Document Info:**")
            st.markdown(f"**File:** {st.session_state.filename}")
            st.markdown(f"**Pages:** {total_pages}")
            
            st.divider()
            
            st.markdown("**🔍 Page Selection**")
            st.session_state.preview_mode = st.radio(
                "Context:", ["Full Document", "Page Range"]
            )

            if st.session_state.preview_mode == "Page Range":
                col1, col2 = st.columns(2)
                with col1:
                    start = st.number_input("Start", min_value=1, max_value=total_pages, value=1)
                with col2:
                    end = st.number_input("End", min_value=start, max_value=total_pages, value=min(total_pages, start + 9))
                st.session_state.start_page = start - 1
                st.session_state.end_page = end - 1
            else:
                st.session_state.start_page = 0
                st.session_state.end_page = total_pages - 1

            if not st.session_state.show_chat:
                if st.button("💬 Start Chat With Your PDF"):
                    st.session_state.show_chat = True
                    st.rerun()

# --- Main Content Area ---
if uploaded_file and st.session_state.pdf_processed:
    if not st.session_state.show_chat:
        st.markdown('<h2 class="sub-header">📄 Document Ready</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">Your document <strong>{st.session_state.filename}</strong> ({st.session_state.total_pages} pages) is loaded. Click "Start Chat With Your PDF" in the sidebar to begin.</div>', unsafe_allow_html=True)
    else:
        # Prompt for user name before starting chat
        if not st.session_state.name_prompted:
            st.markdown('<h2 class="sub-header">👋 Welcome to Knowra!</h2>', unsafe_allow_html=True)
            user_name = st.text_input("Enter your name to start chatting:", placeholder="Your name")
            if st.button("Start Chatting"):
                if user_name.strip():
                    st.session_state.user_name = user_name.strip()
                    st.session_state.name_prompted = True
                    range_text = f"pages {st.session_state.start_page + 1} to {st.session_state.end_page + 1}" if st.session_state.preview_mode == "Page Range" else "the entire document"
                    welcome_msg = f"Hello {st.session_state.user_name}! 👋 I'm Knowra, ready to help you explore {st.session_state.filename} ({range_text}). What would you like to know?"
                    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                    st.rerun()
                else:
                    st.error("Please enter your name to continue.")
        else:
            # Chat Interface
            st.markdown('<h4 class="sub-header">💬 Chat with Your Document</h4>', unsafe_allow_html=True)
            
            range_text = f"pages {st.session_state.start_page + 1} to {st.session_state.end_page + 1}" if st.session_state.preview_mode == "Page Range" else f"entire document ({st.session_state.total_pages} pages)"
            st.markdown(f'<p class="info-box"><strong>{st.session_state.filename}</strong> • {range_text}</p>', unsafe_allow_html=True)

            # Display chat history
            for message in st.session_state.chat_history:
                timestamp = datetime.now().strftime("%H:%M")
                
                if message["role"] == "user":
                    st.markdown(f'''
                    <div class="message-wrapper user-message-wrapper">
                        <div class="message-bubble user-message">
                            <div class="message-header">
                                <span>👤</span>
                                <span>{st.session_state.user_name}</span>
                            </div>
                            <div>{message["content"]}</div>
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="message-wrapper assistant-message-wrapper">
                        <div class="message-bubble assistant-message">
                            <div class="message-header">
                                <span>🤖</span>
                                <span>Knowra</span>
                            </div>
                            <div>{message["content"]}</div>
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

            # Chat input
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_button = st.columns([4, 1])
                with col_input:
                    query = st.text_input("Message", placeholder="Ask me anything about your document...", label_visibility="collapsed")
                with col_button:
                    send_button = st.form_submit_button("Send", use_container_width=True)
                
                if send_button and query:
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    
                    with st.spinner("Thinking..."):
                        try:
                            normalized_question = query.lower().strip().rstrip("!?.")
                            greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
                            if normalized_question in greetings:
                                response = "Hello! I'm ready to answer questions about your document. What would you like to know?"
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            else:
                                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 7})
                                filtered_retriever = PageFilteredRetriever(
                                    base_retriever=retriever,
                                    start_page=st.session_state.start_page,
                                    end_page=st.session_state.end_page
                                )
                                full_chain = create_retrieval_chain(filtered_retriever, st.session_state.document_chain)
                                response = full_chain.invoke({"input": query})
                                answer = response.get("answer", "An error occurred during processing.")
                                context_docs = response.get("context", [])
                                
                                not_found_phrases = [
                                    "not found", 
                                    "cannot answer", 
                                    "do not contain the answer", 
                                    "not in the provided context",
                                    "i cannot answer"
                                ]
                                if any(phrase in answer.lower() for phrase in not_found_phrases):
                                    answer = "I can only answer questions based on the content of the document you provided. Please ask something related to the PDF."
                                    sources = []
                                else:
                                    sources = sorted(list(set(doc.metadata.get("page", -1) + 1 for doc in context_docs)))
                                
                                full_response = f"{answer}<br><br><small>📄 Sources: Page {', '.join(map(str, sources)) if sources else 'N/A'}</small>"
                                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
                    st.rerun()

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.session_state.name_prompted = False
                    st.session_state.user_name = ""
                    st.rerun()
            with col2:
                if st.button("📖 Back to Document"):
                    st.session_state.show_chat = False
                    st.rerun()

else:
    # Welcome screen
    st.markdown('''
<div class="info-box">
    <h3>🚀 Getting Started</h3>
    <p><strong>1.</strong> Upload your PDF using the sidebar</p>
    <p><strong>2.</strong> Choose to analyze the full document or select specific pages</p>
    <p><strong>3.</strong> Start chatting with your document!</p>
    <br>
    <p><strong>⚠️ Note:</strong> Only text-based PDFs are supported for preview. Image-only/scanned PDFs are not supported currently.</p>
    <p>💡 <strong>Tip:</strong> For better performance with large PDFs, select only the pages you need.</p>
</div>
''', unsafe_allow_html=True)