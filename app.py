import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("Chat with PDF")
# --------------------------------------------------
# API Key
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

@st.cache_resource
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )


# --------------------------------------------------
# PDF Extraction
# --------------------------------------------------

def read_pdf(pdf_file):
    documents = []

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number}
                )
            )

    return documents


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# FAISS Vector Store
# --------------------------------------------------

def build_vector_store(documents):
    embeddings = create_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


# --------------------------------------------------
# Gemini LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )


# --------------------------------------------------
# Prompt
# --------------------------------------------------

def get_prompt():
    prompt_template = """
You are an AI assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context, say exactly:

"THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT."

Give the answer in clear bullet points.
Each bullet point should not exceed 200 words.
Explain the answer in a way that a Class 10 student can understand.

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

def answer_question(question, vector_store):
    docs = vector_store.similarity_search(
        question,
        k=10
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = get_prompt()
    llm = get_llm()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    if isinstance(response.content, str):
        return response.content

    return response.content[0]["text"]


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    documents = read_pdf(uploaded_file)

    if not documents:
        st.error("Could not extract text from this PDF.")
        st.stop()

    chunks = split_documents(documents)

    vector_store = build_vector_store(chunks)

    question = st.text_input(
        "Ask a question"
    )
    answer_box = st.empty()
    if question.strip():

        answer = answer_question(
            question,
            vector_store
        )
        answer_box.write(answer)
    else:
        answer_box.empty()

