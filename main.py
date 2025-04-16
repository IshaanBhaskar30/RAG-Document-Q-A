import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

st.set_page_config(page_title="RAG Q&A with Groq + LLaMA3", layout="wide")
st.title("📚 RAG Document Q&A with Groq + LLaMA3")

# 🔐 Ask for API keys
groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")
hf_token = st.text_input("🧠 Enter your Hugging Face Token:", type="password")

# ✅ Continue only when both keys are provided
if groq_api_key and hf_token:
    os.environ['HF_TOKEN'] = hf_token

    # Initialize LLM and embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Define Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.

        <context>
        {context}
        <context>

        Question: {input}
        """
    )

    # Create vector DB from PDF documents
    def create_vector_embedding():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = embeddings
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Path to PDF folder
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )

    # UI to embed documents
    if st.button("📦 Build Document Embeddings"):
        create_vector_embedding()
        st.success("✅ Vector database created from PDFs!")

    # User prompt
    user_prompt = st.text_input("💬 Ask a question from your documents:")

    if user_prompt:
        if "vectors" not in st.session_state:
            st.warning("Please build the document embeddings first.")
        else:
            # RAG chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Answer query
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start

            st.markdown(f"**🧠 Answer:** {response['answer']}")
            st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

            with st.expander("📄 Document Similarity Search Results"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("-----")
else:
    st.info("Please enter your Groq and Hugging Face API keys to begin.")
