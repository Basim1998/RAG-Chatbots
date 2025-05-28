import streamlit as st
from dotenv import load_dotenv

def setup_page_config():
    st.set_page_config(
        page_title="Enhanced In-Vehicle Infotainment RAG Chatbot",
        page_icon="🎵",
        layout="wide"
    )
    st.title("🎵 Enhanced In-Vehicle Infotainment RAG Chatbot")
    st.markdown("""
    This enhanced chatbot provides detailed answers about in-vehicle infotainment systems:
    - **Simple Chatbot**: Uses the Groq API LLM to generate responses based on its knowledge.
    - **RAG Chatbot**: Retrieves relevant context from a knowledge base to generate precise responses.
    - **Advanced RAG Chatbot**: Enhanced with iterative refinement, sentence window retrieval, and chat history context.
    - **Hybrid RAG Chatbot**: Combines semantic (vector) and syntactic (BM25) search with sentence window retrieval.
    - **Agentic RAG Chatbot**: Uses specialized agents for different infotainment domains with dynamic routing and web search.
    Ask about amplifiers, subwoofers, touchscreens, connectivity, or any infotainment-related topic!
    """)

def get_chatbot_settings():
    print("Calling get_chatbot_settings")  # Debug
    load_dotenv()
    
    try:
        from llama_index.core.node_parser import SentenceWindowNodeParser
        sentence_window_available = True
    except ImportError:
        sentence_window_available = False
    
    st.sidebar.header("Chatbot Settings")
    chatbot_mode = st.sidebar.selectbox(
        "Select Chatbot Mode",
        ["Simple Chatbot", "RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"],
        index=4,
        key="chatbot_mode_select"
    )
    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        index=0,
        key="llm_model_select"
    )
    
    settings = {
        'chatbot_mode': chatbot_mode,
        'llm_model': llm_model,
        'sentence_window_available': sentence_window_available,
        'top_k': 8,
        'similarity_cutoff': 0.6,
        'temperature': 0.1,
        'semantic_weight': 0.7,
        'syntactic_weight': 0.3,
        'use_query_rewriting': False,
        'use_reranking': False,
        'use_iterative_refinement': False,
        'use_sentence_window': False,
        'use_chat_history': True,
        'max_refinement_iterations': 3,
        'chunk_size': 512,
        'chunk_overlap': 50,
        'sentence_window_size': 3,
        'show_agent_reasoning': False,
        'enable_multi_agent': False,
        'enable_web_search': False,
        'max_web_results': 3
    }
    
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        settings['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="temperature_slider")
        
        if chatbot_mode in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
            settings['top_k'] = st.slider("Number of chunks to retrieve", 1, 20, 8, key="top_k_slider")
            settings['similarity_cutoff'] = st.slider("Similarity cutoff", 0.0, 1.0, 0.6, 0.05, key="similarity_cutoff_slider")
        
        if chatbot_mode == "Hybrid RAG Chatbot":
            st.subheader("🔄 Hybrid Search Settings")
            settings['semantic_weight'] = st.slider("Semantic Search Weight", 0.0, 1.0, 0.7, 0.1, key="semantic_weight_slider")
            settings['syntactic_weight'] = st.slider("Syntactic Search Weight", 0.0, 1.0, 0.3, 0.1, key="syntactic_weight_slider")
            st.info(f"Total Weight: {settings['semantic_weight'] + settings['syntactic_weight']:.1f}")
        
        if chatbot_mode in ["Advanced RAG Chatbot", "Hybrid RAG Chatbot"]:
            st.subheader("🚀 Advanced Features")
            settings['use_query_rewriting'] = st.checkbox("Query Rewriting", True, key="query_rewriting_checkbox")
            settings['use_reranking'] = st.checkbox("Re-ranking", True, key="reranking_checkbox")
            settings['use_iterative_refinement'] = st.checkbox("Iterative Refinement", True, key="iterative_refinement_checkbox")
            settings['use_sentence_window'] = st.checkbox("Sentence Window Retrieval", sentence_window_available, key="sentence_window_checkbox")
            settings['use_chat_history'] = st.checkbox("Chat History Context", True, key="chat_history_checkbox")
            settings['max_refinement_iterations'] = st.slider("Max Refinement Iterations", 1, 5, 3, key="max_refinement_iterations_slider")
            settings['chunk_size'] = st.slider("Chunk Size", 128, 1024, 512, key="chunk_size_slider")
            settings['chunk_overlap'] = st.slider("Chunk Overlap", 0, 200, 50, key="chunk_overlap_slider")
            if sentence_window_available:
                settings['sentence_window_size'] = st.slider("Sentence Window Size", 1, 5, 3, key="sentence_window_size_slider")
        
        if chatbot_mode == "Agentic RAG Chatbot":
            st.subheader("🤖 Agentic RAG Options")
            settings['show_agent_reasoning'] = st.checkbox("Show Agent Reasoning", True, key="show_agent_reasoning_checkbox")
            settings['enable_multi_agent'] = st.checkbox("Enable Multi-Agent Collaboration", True, key="enable_multi_agent_checkbox")
            settings['enable_web_search'] = st.checkbox("Enable Web Search", True, key="enable_web_search_checkbox")
            settings['max_web_results'] = st.slider("Max Web Results", 1, 10, 3, key="max_web_results_slider")
            settings['use_sentence_window'] = st.checkbox("Sentence Window Retrieval", sentence_window_available, key="agent_sentence_window_checkbox")
            if sentence_window_available:
                settings['sentence_window_size'] = st.slider("Sentence Window Size", 1, 5, 3, key="agent_sentence_window_size_slider")
            settings['use_chat_history'] = True
            settings['use_query_rewriting'] = False
            settings['use_reranking'] = False
            settings['use_iterative_refinement'] = False
    
    return settings