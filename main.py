import streamlit as st
from config import setup_page_config, get_chatbot_settings
from components.rag_components import load_rag_components
from components.query_engine import get_enhanced_query_engine, simple_chatbot_query_enhanced
from components.agents import agentic_rag_query

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "settings" not in st.session_state:
        st.session_state.settings = {}

def display_chat_history(show_agent_reasoning):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "reasoning" in message and message["reasoning"] and show_agent_reasoning:
                with st.expander("🧠 Agent Reasoning"):
                    st.markdown(message["reasoning"])
            if "web_sources" in message and message["web_sources"]:
                with st.expander("🌐 Web Search Results"):
                    for i, source in enumerate(message["web_sources"]):
                        st.markdown(f"**Web Source {i+1}:** [{source['title']}]({source['href']})")
                        st.markdown(f"**Content:** {source['body'][:300]}...")
                        st.markdown(f"**Provider:** {source['source']}")
                        st.markdown("---")
            if "source_nodes" in message and message["source_nodes"]:
                with st.expander("📚 View Source Documents"):
                    for i, source in enumerate(message["source_nodes"]):
                        text = source.text[:500] + "..." if len(source.text) > 500 else source.text
                        score = source.score if hasattr(source, 'score') else 0
                        sources_info = ""
                        if hasattr(source, 'node') and hasattr(source.node, 'metadata'):
                            retrieval_sources = source.node.metadata.get('retrieval_sources', [])
                            if retrieval_sources:
                                sources_info = f" (Retrieved via: {', '.join(retrieval_sources)})"
                        st.markdown(f"**Source {i+1}:** {text}")
                        st.markdown(f"**Relevance Score:** {score:.4f}{sources_info}")
                        st.markdown("---")

def display_active_features(chatbot_mode, settings):
    if chatbot_mode in ["Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
        with st.expander("🚀 Enhanced Features Active"):
            features = []
            if chatbot_mode == "Hybrid RAG Chatbot":
                features.append("✅ Hybrid Search (Semantic + Syntactic)")
                features.append(f"✅ Semantic Weight: {settings['semantic_weight']:.1f}")
                features.append(f"✅ Syntactic Weight: {settings['syntactic_weight']:.1f}")
            elif chatbot_mode == "Agentic RAG Chatbot":
                features.append("✅ Specialized Agent Routing")
                if settings['enable_multi_agent']:
                    features.append("✅ Multi-Agent Collaboration")
                if settings['enable_web_search']:
                    features.append("✅ Web Search (DuckDuckGo)")
                if settings['show_agent_reasoning']:
                    features.append("✅ Agent Reasoning Display")
            if settings['use_iterative_refinement']:
                features.append("✅ Iterative Refinement")
            if settings['use_sentence_window'] and settings['sentence_window_available']:
                features.append("✅ Sentence Window Retrieval")
            if settings['use_chat_history']:
                features.append("✅ Chat History Context")
            if settings['use_query_rewriting']:
                features.append("✅ Query Rewriting")
            if settings['use_reranking']:
                features.append("✅ Advanced Re-ranking")
            st.markdown('\n'.join(features))

def main():
    try:
        setup_page_config()
        initialize_session_state()
        
        # Render sidebar exactly once per run
        with st.sidebar:
            print("Rendering sidebar in main.py")  # Debug
            settings = get_chatbot_settings()
            st.session_state.settings.update(settings)  # Update session state
        
        index, llm, bm25_retriever = load_rag_components(st.session_state.settings)
        
        if st.session_state.settings['chatbot_mode'] in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot"] and index is None:
            st.error("Knowledge base failed to load. Please ensure the './data' directory contains documents and try again.")
            return
        
        display_active_features(st.session_state.settings['chatbot_mode'], st.session_state.settings)
        display_chat_history(st.session_state.settings['show_agent_reasoning'])
        
        user_query = st.chat_input("Ask about in-vehicle infotainment systems...", key="chat_input")
        
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                spinner_text = f"Processing with {st.session_state.settings['chatbot_mode']}..."
                if st.session_state.settings['chatbot_mode'] == "Agentic RAG Chatbot":
                    spinner_text = "Agents analyzing and searching web..."
                with st.spinner(spinner_text):
                    if st.session_state.settings['chatbot_mode'] == "Simple Chatbot":
                        response_text, source_nodes = simple_chatbot_query_enhanced(user_query, llm, st.session_state.settings)
                        web_sources = []
                        reasoning_text = ""
                    elif st.session_state.settings['chatbot_mode'] == "Agentic RAG Chatbot":
                        response_text, source_nodes, web_sources, reasoning_text = agentic_rag_query(user_query, llm, index, st.session_state.settings)
                    else:
                        query_engine = get_enhanced_query_engine(index, llm, bm25_retriever, st.session_state.settings)
                        response = query_engine.query(user_query)
                        response_text = str(response)
                        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                        web_sources = []
                        reasoning_text = ""
                    
                    message_placeholder.markdown(response_text)
                    if reasoning_text and st.session_state.settings['show_agent_reasoning'] and st.session_state.settings['chatbot_mode'] == "Agentic RAG Chatbot":
                        with st.expander("🧠 Agent Reasoning"):
                            st.markdown(reasoning_text)
                    if web_sources:
                        with st.expander("🌐 Web Search Results"):
                            for i, source in enumerate(web_sources):
                                st.markdown(f"**Web Source {i+1}:** [{source['title']}]({source['href']})")
                                st.markdown(f"**Content:** {source['body'][:300]}...")
                                st.markdown(f"**Provider:** {source['source']}")
                                st.markdown("---")
                    if source_nodes and st.session_state.settings['chatbot_mode'] in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
                        with st.expander("📚 View Source Documents"):
                            for i, source in enumerate(source_nodes):
                                text = source.text[:500] + "..." if len(source.text) > 500 else source.text
                                score = source.score if hasattr(source, 'score') else 0
                                sources_info = ""
                                if hasattr(source, 'node') and hasattr(source.node, 'metadata'):
                                    retrieval_sources = source.node.metadata.get('retrieval_sources', [])
                                    if retrieval_sources:
                                        sources_info = f" (Retrieved via: {', '.join(retrieval_sources)})"
                                st.markdown(f"**Source {i+1}:** {text}")
                                st.markdown(f"**Relevance Score:** {score:.4f}{sources_info}")
                                st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "source_nodes": source_nodes,
                "web_sources": web_sources,
                "reasoning": reasoning_text
            })
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your configuration and ensure all required components are properly set up.")

if __name__ == "__main__":
    main()