from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone
load_dotenv()


def load_pdf(pdf_path):
    loader=DirectoryLoader(pdf_path,glob='*.pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    return document

