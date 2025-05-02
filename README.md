📚 RAG-Based Document Q&A with Groq + LLaMA3

This project is an advanced Streamlit application that implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions based on the contents of PDF documents. The app uses Groq’s ultra-fast LLaMA3-8B-8192 model for generating responses and Hugging Face’s MiniLM model for semantic embeddings. It is designed to read and understand documents (like research papers) and provide accurate, context-specific answers to user queries—all within a fast, interactive UI.

The workflow involves loading PDF documents (in this case, two sample papers: “Attention Is All You Need” and one on Large Language Models), splitting their contents into manageable text chunks, embedding them using Hugging Face transformers, and storing them in a FAISS vector database for similarity-based retrieval. When a user asks a question, the app retrieves the most relevant content chunks and feeds them to the LLM along with the query, ensuring that the answer remains grounded in the source material.

Key technologies used:

->Groq API + LLaMA3 for blazing-fast inference

->LangChain for document loading, chunking, and retrieval chain construction

->FAISS for vector storage and similarity search

->Hugging Face Embeddings (MiniLM) for encoding document chunks

->Streamlit for a modern, reactive user interface

This tool is ideal for:

->Researchers and students wanting to query papers without reading them in full

->Developers or analysts building search-augmented knowledge assistants

->Anyone interested in combining RAG, vector databases, and LLMs in a real-world application
