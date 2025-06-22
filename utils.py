import io
from typing import List
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

def get_page_count(pdf_bytes: bytes) -> int:
    """Get the number of pages in a PDF."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = pdf_document.page_count
        pdf_document.close()
        return page_count
    except Exception:
        return 0

def load_and_split_pdf(pdf_stream: io.BytesIO, filename: str) -> List[Document]:
    """Load and split PDF into documents with page metadata."""
    try:
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        documents = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text("text")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "filename": filename}
                ))
        pdf_document.close()
        return documents
    except Exception:
        return []

def create_vector_store(documents: List[Document]) -> FAISS:
    """Create FAISS vector store from documents using locally stored embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_rag_chain():
    """Create the core part of the RAG chain (prompt + LLM)."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain the answer, state clearly that the answer is not found in the provided pages.
Do not use any external knowledge. Be concise.

Context:
{context}

Question: {input}

Answer:"""
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

class PageFilteredRetriever(BaseRetriever):
    """Custom retriever to filter documents by page range."""
    base_retriever: BaseRetriever
    start_page: int
    end_page: int

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self.base_retriever.aget_relevant_documents(query)
        return [doc for doc in docs if self.start_page <= doc.metadata.get('page', -1) <= self.end_page]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        return [doc for doc in docs if self.start_page <= doc.metadata.get('page', -1) <= self.end_page]