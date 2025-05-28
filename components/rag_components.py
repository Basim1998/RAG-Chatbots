import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from langchain_groq import ChatGroq
from components.retrieval import initialize_bm25

@st.cache_resource
def load_rag_components(settings):
    langchain_llm = ChatGroq(
        model=settings['llm_model'],
        temperature=settings['temperature'],
        max_retries=2,
    )
    llm = LangChainLLM(llm=langchain_llm)
    Settings.llm = llm
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    persist_dir = "./storage_sentence_window" if settings['use_sentence_window'] and settings['sentence_window_available'] else "./storage"
    index = None
    bm25_retriever = None
    
    if settings['chatbot_mode'] in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
        try:
            if os.path.exists(persist_dir):
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)
            else:
                # Check if data directory exists and has files
                if not os.path.exists("./data") or not os.listdir("./data"):
                    st.error("Data directory './data' is missing or empty. Please add documents to './data'.")
                    return None, llm, None
                
                documents = SimpleDirectoryReader("./data").load_data()
                storage_context = StorageContext.from_defaults()
                
                if settings['use_sentence_window'] and settings['sentence_window_available']:
                    from llama_index.core.node_parser import SentenceWindowNodeParser
                    node_parser = SentenceWindowNodeParser.from_defaults(
                        window_size=settings['sentence_window_size'],
                        window_metadata_key="window",
                        original_text_metadata_key="original_text"
                    )
                    nodes = node_parser.get_nodes_from_documents(documents)
                else:
                    nodes = documents  # Use default node parsing
                
                index = VectorStoreIndex(nodes, storage_context=storage_context)
                index.storage_context.persist(persist_dir=persist_dir)
        except Exception as e:
            st.error(f"Failed to load or create knowledge base: {str(e)}")
            return None, llm, None
        
        if settings['chatbot_mode'] == "Hybrid RAG Chatbot" and index is not None:
            bm25_retriever = initialize_bm25(index)
    
    return index, llm, bm25_retriever