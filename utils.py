import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import streamlit as st

def process_pdf(uploaded_file):
    """
    Process an uploaded PDF file: save to temp, load, split, and return documents.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40
        )
        docs = text_splitter.split_documents(pages)

        lc_docs = [
            Document(
                page_content=doc.page_content.replace("\n", ""),
                metadata={'source': uploaded_file.name}
            ) for doc in docs
        ]
        return lc_docs
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
