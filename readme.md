main.py: Entry point for the Streamlit app, handling UI and user interactions.
config.py: Configuration settings and environment variable loading.
components/rag_components.py: Logic for loading RAG components (LLM, index, BM25).
components/agents.py: Agent-related classes (InfotainmentAgent, WebResearchAgent, AgentCoordinator).
components/retrieval.py: Retrieval logic (SimpleBM25, hybrid retrieval, query rewriting).
components/query_engine.py: Enhanced query engine and response generation logic.
components/web_search.py: Web search tool for external data retrieval.
utils/helpers.py: Utility functions for chat history and other helpers.