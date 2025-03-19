import streamlit as st
import os
import pandas as pd
import shutil
import time
import re
from typing import List, Dict, Any, Tuple
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Use more modern document loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Persistent directory for Chroma
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"

# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize or load vector store
def initialize_vector_store(force_reset=False):
    if force_reset and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            vector_store._collection.count()
            return vector_store
        except Exception as e:
            st.warning(f"Failed to load vector store: {str(e)}. Resetting directory.")
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
    
    return None

# Streamlit UI
st.title("Local free, secure and personalized Generative AI RAG assistant")

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "docx", "xlsx"], accept_multiple_files=True)

def process_file(file):
    """Process a file and return LangChain documents with correct metadata."""
    temp_file = f"temp_file_{file.name}"
    with open(temp_file, "wb") as f:
        f.write(file.getbuffer())
    
    original_filename = file.name
    file_type = original_filename.split(".")[-1].lower()
    
    try:
        if file_type == "txt":
            loader = TextLoader(temp_file)
            documents = loader.load()
        elif file_type == "pdf":
            loader = PyPDFLoader(temp_file)
            documents = loader.load()
        elif file_type == "docx":
            loader = Docx2txtLoader(temp_file)
            documents = loader.load()
        elif file_type == "xlsx":
            loader = UnstructuredExcelLoader(temp_file)
            documents = loader.load()
        
        # Enhance metadata
        for doc in documents:
            # Keep original text case for better matching
            doc.metadata["source"] = original_filename
            doc.metadata["file_type"] = file_type
            # Add a unique id for each chunk to aid in traceability
            doc.metadata["chunk_id"] = f"{original_filename}_{id(doc)}"
        
        os.remove(temp_file)
        return documents
    except Exception as e:
        os.remove(temp_file)
        raise e

# Process uploaded files
if uploaded_files:
    # Use a more semantic chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks to maintain more context
        chunk_overlap=200,  # Substantial overlap to avoid losing context at boundaries
        separators=["\n\n", "\n", ". ", " ", ""],  # Respect paragraph and sentence boundaries
        length_function=len
    )
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                documents = process_file(uploaded_file)
                chunks = text_splitter.split_documents(documents)
                
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        collection_name=COLLECTION_NAME,
                        persist_directory=CHROMA_PERSIST_DIR
                    )
                    st.success(f"Created new vector store with '{uploaded_file.name}'")
                else:
                    st.session_state.vector_store.add_documents(chunks)
                    st.success(f"Added '{uploaded_file.name}' to vector store")
                
                st.session_state.processed_files.add(uploaded_file.name)
            except Exception as e:
                st.error(f"Error processing '{uploaded_file.name}': {str(e)}")

# Display processed files
if st.session_state.processed_files:
    st.subheader("Processed Files")
    for file_name in st.session_state.processed_files:
        st.write(f"- {file_name}")

# Question input
st.subheader("Ask a Question")
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Ask a question about the documents:")
    submit_button = st.form_submit_button(label="Submit")

if submit_button and question:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = initialize_vector_store()
    
    if st.session_state.vector_store:
        llm = OllamaLLM(model="llama3")
        
        # Use MMR retrieval for more diverse and relevant results
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 8,  # Retrieve more documents for better coverage
                "fetch_k": 20,  # Consider more candidates for diversity
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        # Debug retrieved documents
        with st.expander("Debug Information"):
            count = st.session_state.vector_store._collection.count()
            st.write(f"Total chunks in vector store: {count}")
            retrieved_docs = retriever.invoke(question)
            st.write("Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", "Unknown ID")
                st.write(f"Doc {i+1} from '{source}' (ID: {chunk_id}): {doc.page_content[:200]}...")

        # Enhance prompt with clear source attribution requirements
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a precise assistant answering based solely on the provided document content. "
                "Extract the answer directly from the context below. If the context doesn't contain the answer, respond only with 'Unable to answer based on documents.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer with specific reference to the source documents. When quoting information, clearly indicate which specific document it came from. "
                "Use the format 'According to [document name]' when directly citing information. "
                "If multiple documents contain similar information, mention all relevant sources."
            )
        )

        # Implement document tracking for better attribution
        def format_docs_with_sources(docs: List[LangChainDocument]) -> str:
            """Format documents with clear source markers for better attribution."""
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
                
                # Add source markers that can be traced in the final answer
                formatted_text = f"[Source: {source} | ID: {chunk_id}]\n{doc.page_content}\n[End of source: {source}]"
                formatted_docs.append(formatted_text)
            
            return "\n\n".join(formatted_docs)

        # Enhanced RAG chain with better source tracking
        try:
            # Use LCEL (LangChain Expression Language) for more flexibility and control
            rag_chain = (
                {"context": lambda x: format_docs_with_sources(retriever.invoke(x["query"])), 
                 "question": lambda x: x["query"]}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("Generating answer..."):
                answer = rag_chain.invoke({"query": question})
                
                # Extract source references from the answer
                source_pattern = r"according to [\[\(]?([^\]\)\[:]+)[\]\)]?"
                mentioned_sources = re.findall(source_pattern, answer.lower())
                
                # Enhance source detection with more sophisticated pattern matching
                def extract_contributing_sources(answer: str, docs: List[LangChainDocument]) -> List[str]:
                    """Extract contributing sources using multiple heuristics."""
                    contributing_sources = set()
                    
                    # Extract directly mentioned sources
                    source_patterns = [
                        r"according to [\[\(]?([^\]\)\[:]+)[\]\)]?",
                        r"from [\[\(]?([^\]\)\[:]+)[\]\)]?",
                        r"in [\[\(]?([^\]\)\[:]+)[\]\)]?",
                        r"cited in [\[\(]?([^\]\)\[:]+)[\]\)]?"
                    ]
                    
                    for pattern in source_patterns:
                        for match in re.finditer(pattern, answer.lower()):
                            source_mention = match.group(1).strip()
                            # Find the closest matching actual source
                            for doc in docs:
                                doc_source = doc.metadata.get("source", "").lower()
                                if source_mention in doc_source or doc_source in source_mention:
                                    contributing_sources.add(doc.metadata.get("source", "Unknown"))
                    
                    # If no sources found through explicit mention, use semantic matching
                    if not contributing_sources:
                        # Extract key sentences from the answer
                        answer_sentences = re.split(r'(?<=[.!?])\s+', answer)
                        for sentence in answer_sentences:
                            if len(sentence) < 10:  # Skip very short sentences
                                continue
                                
                            # Check if sentence content appears in any document
                            sentence_lower = sentence.lower()
                            for doc in docs:
                                doc_content = doc.page_content.lower()
                                # Use sliding window for partial matches
                                words = sentence_lower.split()
                                for i in range(len(words) - 3):  # At least 4 consecutive words
                                    phrase = " ".join(words[i:i+4])
                                    if phrase in doc_content and len(phrase) > 15:  # Significant phrase
                                        contributing_sources.add(doc.metadata.get("source", "Unknown"))
                    
                    return list(contributing_sources)
                
                # Get retrieved documents for source extraction
                retrieved_docs = retriever.invoke(question)
                contributing_sources = extract_contributing_sources(answer, retrieved_docs)
                
                # Debug source extraction
                with st.expander("Source Attribution Debug"):
                    st.write(f"Raw mentioned sources: {mentioned_sources}")
                    st.write(f"Final contributing sources: {contributing_sources}")
                    
                    # Show sentence-level matching
                    st.write("Sentence-level matching:")
                    answer_sentences = re.split(r'(?<=[.!?])\s+', answer)
                    for i, sentence in enumerate(answer_sentences):
                        if len(sentence) < 10:
                            continue
                        st.write(f"Sentence {i+1}: {sentence}")
                        matching_sources = []
                        for doc in retrieved_docs:
                            if any(phrase in doc.page_content.lower() for phrase in [s.lower() for s in sentence.split('.') if len(s) > 15]):
                                matching_sources.append(doc.metadata.get("source", "Unknown"))
                        st.write(f"  - Matching sources: {matching_sources}")

                history_entry = {
                    "question": question,
                    "answer": answer,
                    "sources": contributing_sources
                }

                st.session_state.qa_history.append(history_entry)
        except Exception as e:
            st.error(f"Error initializing QA chain or generating answer: {str(e)}")
    else:
        st.error("No documents loaded. Please upload files first.")

# Display history
st.subheader("Question and Answer History")
for i, qa in enumerate(st.session_state.qa_history):
    st.write(f"**Q{i+1}:** {qa['question']}")
    st.write(f"**A{i+1}:** {qa['answer']}")
    if qa["sources"]:
        st.write(f"**Sources:** {', '.join(qa['sources'])}")
    else:
        st.write("**Sources:** None")
    st.write("---")

# Reset button
if st.button("Reset Knowledge Base"):
    try:
        st.session_state.vector_store = initialize_vector_store(force_reset=True)
        st.session_state.processed_files = set()
        st.session_state.qa_history = []
        time.sleep(1)
        st.success("Knowledge base reset successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error resetting: {str(e)}")
