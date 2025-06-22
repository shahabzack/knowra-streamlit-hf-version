# Knowra — Your Aura of Answers 🧠📄

**Knowra** is an intelligent PDF chatbot that allows users to upload text-based PDF documents and ask questions about their content.  
Powered by a Retrieval-Augmented Generation (RAG) pipeline, Knowra extracts text from PDFs, stores it in a FAISS vector database, and uses Google’s Gemini LLM to provide accurate, context-aware answers — with page references.

---

## ⚙️ Why This Version?

The original Knowra app used **FastAPI** and was hosted on **Render**, but:

- 🚫 Render’s free plan had **memory limitations** that caused issues during PDF processing.
- 🚫 Hugging Face Spaces does **not support FastAPI** directly.

So this version was created specifically for **Hugging Face**:

- ✅ Replaced FastAPI with **Streamlit**
- ✅ Simplified frontend-only app
- ✅ Fully compatible with Hugging Face Spaces
- ✅ Great for demoing and lightweight deployment

---

## 🧩 Features

- 📄 Upload and read **text-based PDFs**
- ❓ Ask questions in plain English
- 🔍 Uses **FAISS** for semantic vector search
- 🤖 Answers powered by **Google Gemini (Generative AI)**
- 📍 Includes **page references** to support traceable answers

---

## 🔗 Original Full-Stack Version

For the full-featured version (FastAPI backend + client), advanced configuration, and deployment via Render or local hosting, check the original repository:

👉 [Knowra RAG Assistant (Main GitHub Repo)](https://github.com/shahabzack/Knowra-rag-assistant)

---

## 🚀 Live Demo (Streamlit + Hugging Face)

This version is hosted on Hugging Face Spaces using **Streamlit** for the frontend. It’s optimized for free-tier deployment and quick PDF interactions.

---

## 👤 Connect with Me

Created by [Shahab Zack](https://shahabzack.github.io/Ds_portfolio/)  
Feel free to connect, collaborate, or share feedback!

---

## 📄 License

MIT License — see [`LICENSE`](./LICENSE) for details.
