import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

_ = load_dotenv(find_dotenv())


st.set_page_config(
    page_title="Upload Document",
    page_icon="📄",
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None


uploaded_file = st.file_uploader(
    "Choose a PDF file", type=["pdf"]
)

if st.session_state.retriever is not None and uploaded_file is None:
    st.success("You've already uploaded a document!", icon="✅")

if uploaded_file is not None:
    with st.spinner("Loading document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)

    with st.spinner("Splitting document into chunks..."):
        # Split the document into chunks
        # Using RecursiveCharacterTextSplitter to split the document into smaller chunks
        # This is useful for large documents to improve processing speed and accuracy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
            )
        all_splits = text_splitter.split_documents(pages)

    with st.spinner("Creating retriever..."):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(documents=all_splits)
        retriever = vector_store.as_retriever(search_type="mmr")

    st.session_state.retriever = retriever
    st.success("Document uploaded successfully!", icon="✅")

    uploaded_file = None  # Reset the file uploader after processing