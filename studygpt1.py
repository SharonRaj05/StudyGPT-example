import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
import os


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"






# --- Streamlit Page Config ---
st.set_page_config(page_title="StudyGPT - JEE Assistant")

# --- Define the Web Pages to Scrape ---
jee_websites = [
     
    "https://byjus.com/jee/", 
    "https://www.embibe.com/exams/jee-main/",
    "https://www.vedantu.com/jee"
]

@st.cache_resource
def load_web_index():
    documents = []
    for url in jee_websites:
        loader = WebBaseLoader(url)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)
    return db

# --- Initialize LLM and RAG Chain ---
def get_rag_chain():
    db = load_web_index()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        task="text-generation",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa
qa_chain = get_rag_chain()

# --- Streamlit UI ---
st.title("📘 StudyGPT - JEE Study Assistant")
st.write("Ask me any concept-related question from JEE Physics, Chemistry, or Math!")

query = st.text_input("Enter your question here:", "")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.success(answer)

st.markdown("---")
st.markdown("Created with ❤️ for JEE aspirants")

