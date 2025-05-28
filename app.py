import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from langchain_groq import ChatGroq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore, TextNode
import re
from collections import Counter
import math
import time
from duckduckgo_search import DDGS

# Optional imports for enhanced features
try:
    from llama_index.core.node_parser import SentenceWindowNodeParser
    SENTENCE_WINDOW_AVAILABLE = True
except ImportError:
    SENTENCE_WINDOW_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Enhanced In-Vehicle Infotainment RAG Chatbot",
    page_icon="🎵",
    layout="wide"
)

# App title and description
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

# Sidebar for main configuration 
st.sidebar.header("Chatbot Settings")

# Chatbot mode selection
chatbot_mode = st.sidebar.selectbox(
    "Select Chatbot Mode",
    ["Simple Chatbot", "RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"],
    index=4
)

# Model selection
llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0
)

# Advanced settings under expander
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    if chatbot_mode in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
        top_k = st.slider("Number of chunks to retrieve", 1, 20, 8)
        similarity_cutoff = st.slider("Similarity cutoff", 0.0, 1.0, 0.6, 0.05)
    else:
        top_k = 8
        similarity_cutoff = 0.6
    
    # Hybrid RAG specific settings
    if chatbot_mode == "Hybrid RAG Chatbot":
        st.subheader("🔄 Hybrid Search Settings")
        semantic_weight = st.slider("Semantic Search Weight", 0.0, 1.0, 0.7, 0.1)
        syntactic_weight = st.slider("Syntactic Search Weight", 0.0, 1.0, 0.3, 0.1)
        st.info(f"Total Weight: {semantic_weight + syntactic_weight:.1f}")
    
    # Advanced and Hybrid RAG settings
    if chatbot_mode in ["Advanced RAG Chatbot", "Hybrid RAG Chatbot"]:
        st.subheader("🚀 Advanced Features")
        use_query_rewriting = st.checkbox("Query Rewriting", True)
        use_reranking = st.checkbox("Re-ranking", True)
        use_iterative_refinement = st.checkbox("Iterative Refinement", True)
        use_sentence_window = st.checkbox("Sentence Window Retrieval", SENTENCE_WINDOW_AVAILABLE)
        use_chat_history = st.checkbox("Chat History Context", True)
        max_refinement_iterations = st.slider("Max Refinement Iterations", 1, 5, 3)
        chunk_size = st.slider("Chunk Size", 128, 1024, 512)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
        if SENTENCE_WINDOW_AVAILABLE:
            sentence_window_size = st.slider("Sentence Window Size", 1, 5, 3)
        else:
            sentence_window_size = 3
    elif chatbot_mode == "Agentic RAG Chatbot":
        st.subheader("🤖 Agentic RAG Options")
        show_agent_reasoning = st.checkbox("Show Agent Reasoning", True)
        enable_multi_agent = st.checkbox("Enable Multi-Agent Collaboration", True)
        enable_web_search = st.checkbox("Enable Web Search", True)
        max_web_results = st.slider("Max Web Results", 1, 10, 3)
        use_sentence_window = st.checkbox("Sentence Window Retrieval", SENTENCE_WINDOW_AVAILABLE)
        if SENTENCE_WINDOW_AVAILABLE:
            sentence_window_size = st.slider("Sentence Window Size", 1, 5, 3)
        else:
            sentence_window_size = 3
        use_chat_history = True
        use_query_rewriting = False
        use_reranking = False
        use_iterative_refinement = False
        max_refinement_iterations = 3
        chunk_size = 512
        chunk_overlap = 50
        semantic_weight = 0.7
        syntactic_weight = 0.3
    else:
        use_query_rewriting = False
        use_reranking = False
        use_iterative_refinement = False
        use_sentence_window = False
        use_chat_history = True
        max_refinement_iterations = 3
        chunk_size = 512
        chunk_overlap = 50
        sentence_window_size = 3
        semantic_weight = 0.7
        syntactic_weight = 0.3
        show_agent_reasoning = False
        enable_multi_agent = False
        enable_web_search = False
        max_web_results = 3

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.nodes = []
    
    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, nodes):
        self.nodes = nodes
        corpus = []
        for node in nodes:
            tokens = self._tokenize(node.text)
            corpus.append(tokens)
            self.doc_len.append(len(tokens))
        self.corpus = corpus
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        df = {}
        for document in corpus:
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] = df.get(word, 0) + 1
        for word, freq in df.items():
            self.idf[word] = math.log((len(corpus) - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query):
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[i]
            for word in query_tokens:
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    idf = self.idf.get(word, 0)
                    score += idf * (freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
            scores.append(score)
        return scores
    
    def retrieve(self, query, top_k=10):
        scores = self.get_scores(query)
        scored_nodes = []
        for idx, score in enumerate(scores):
            if idx < len(self.nodes):
                node = self.nodes[idx]
                scored_node = NodeWithScore(node=node, score=score)
                scored_nodes.append(scored_node)
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        return scored_nodes[:top_k]

class WebSearchTool:
    def __init__(self):
        self.provider = "duckduckgo"
    
    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            search_query = f"{query} automotive infotainment system car"
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(search_query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'body': result.get('body', ''),
                        'href': result.get('href', ''),
                        'source': 'DuckDuckGo'
                    })
                return results
        except Exception as e:
            st.warning(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        return self.search_duckduckgo(query, max_results)

class InfotainmentAgent:
    def __init__(self, name: str, expertise: str, llm, index, web_search_tool: Optional[WebSearchTool] = None):
        self.name = name
        self.expertise = expertise
        self.llm = llm
        self.index = index
        self.web_search_tool = web_search_tool
        self.query_engine = self._create_query_engine() if index else None
    
    def _create_query_engine(self):
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        if use_sentence_window and SENTENCE_WINDOW_AVAILABLE:
            postprocessors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )
        
        prompt_template = PromptTemplate(
            f"""You are a specialist agent focused on {self.expertise} in automotive infotainment systems.
            Provide a detailed, technical response based on the context provided.
            Focus specifically on {self.expertise} aspects and provide actionable insights.
            
            Query: {{query_str}}
            Context: {{context_str}}
            
            Specialized Response:
            """
        )
        
        # Use ResponseSynthesizer with custom prompt
        response_synthesizer = get_response_synthesizer(
            text_qa_template=prompt_template,
            response_mode="compact"
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors
        )
    
    def can_handle_query(self, query: str) -> float:
        expertise_keywords = {
            "Audio Systems": ["amplifier", "speaker", "subwoofer", "audio", "sound", "music", "equalizer", "bass", "treble"],
            "Display & UI": ["touchscreen", "display", "screen", "interface", "ui", "navigation", "menu", "button"],
            "Connectivity": ["bluetooth", "wifi", "usb", "connectivity", "wireless", "pairing"],
            "Installation": ["installation", "wiring", "mounting", "setup", "configuration"],
            "Web Research": ["latest", "new", "recent", "update", "trend"]
        }
        
        keywords = expertise_keywords.get(self.expertise, [])
        query_lower = query.lower()
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def search_web(self, query: str, max_results: int) -> List[Dict]:
        if not self.web_search_tool or not enable_web_search:
            return []
        domain_query = f"{query} {self.expertise.lower()}"
        return self.web_search_tool.search(domain_query, max_results)
    
    def process_query(self, query: str):
        local_response = None
        web_results = []
        
        if self.query_engine:
            local_response = self.query_engine.query(query)
        
        if enable_web_search:
            web_results = self.search_web(query, max_web_results)
        
        if web_results:
            web_context = "\n\n".join([
                f"**{result['title']}** (Source: {result['source']})\n{result['body'][:300]}..."
                for result in web_results
            ])
            
            combined_prompt = PromptTemplate(
                f"""You are a specialist in {self.expertise} for automotive infotainment systems.
                Provide a comprehensive response using both the local knowledge base and recent web information.
                
                Query: {{query}}
                Local Knowledge: {{local_info}}
                Recent Web Information:
                {{web_info}}
                
                Combined Response:
                """
            )
            local_info = str(local_response) if local_response else "No local information available."
            combined_response = self.llm.complete(
                combined_prompt.format(
                    query=query,
                    local_info=local_info,
                    web_info=web_context
                )
            )
            
            class CombinedResponse:
                def __init__(self, text, source_nodes, web_sources):
                    self.response = text
                    self.source_nodes = source_nodes
                    self.web_sources = web_sources
                
                def __str__(self):
                    return self.response
            
            source_nodes = local_response.source_nodes if local_response and hasattr(local_response, 'source_nodes') else []
            return CombinedResponse(combined_response.text, source_nodes, web_results)
        
        return local_response

class WebResearchAgent(InfotainmentAgent):
    def __init__(self, llm, web_search_tool):
        super().__init__("Web Research Specialist", "Web Research", llm, None, web_search_tool)
    
    def can_handle_query(self, query: str) -> float:
        web_keywords = ["latest", "new", "recent", "update", "market", "trend"]
        query_lower = query.lower()
        matches = sum(1 for keyword in web_keywords if keyword in query_lower)
        return min(matches * 0.5, 1.0)
    
    def process_query(self, query: str):
        web_results = self.search_web(query, max_web_results)
        
        if not web_results:
            return "No recent web information found for this query."
        
        web_context = "\n\n".join([
            f"**{result['title']}** (Source: {result['source']})\n{result['body']}"
            for result in web_results
        ])
        
        research_prompt = PromptTemplate(
            """You are a web research specialist for automotive infotainment systems.
            Analyze the following web search results and provide a comprehensive summary.
            
            Query: {query}
            Web Search Results:
            {web_context}
            
            Provide a detailed analysis based on the web information:
            """
        )
        
        response = self.llm.complete(
            research_prompt.format(query=query, web_context=web_context)
        )
        
        class WebResponse:
            def __init__(self, text, web_sources):
                self.response = text
                self.source_nodes = []
                self.web_sources = web_sources
            
            def __str__(self):
                return self.response
        
        return WebResponse(response.text, web_results)

class AgentCoordinator:
    def __init__(self, agents: List[InfotainmentAgent], llm):
        self.agents = agents
        self.llm = llm
    
    def route_query(self, query: str) -> InfotainmentAgent:
        agent_scores = {}
        for agent in self.agents:
            score = agent.can_handle_query(query)
            agent_scores[agent.name] = score
        
        best_agent = max(self.agents, key=lambda a: agent_scores[a.name])
        return best_agent, agent_scores
    
    def synthesize_response(self, query: str, responses: Dict[str, Any]) -> str:
        synthesis_prompt = PromptTemplate(
            """You are a master coordinator for automotive infotainment systems.
            Synthesize the following specialist responses into a comprehensive, coherent answer.
            
            Original Query: {query}
            
            Specialist Responses:
            {responses}
            
            Provide a structured, comprehensive response that integrates the specialist knowledge:
            """
        )
        
        response_text = "\n\n".join([f"**{name}**: {resp}" for name, resp in responses.items()])
        prompt = synthesis_prompt.format(
            query=query,
            responses=response_text
        )
        
        synthesis = self.llm.complete(prompt)
        return synthesis.text

@st.cache_resource
def load_rag_components():
    langchain_llm = ChatGroq(
        model=llm_model,
        temperature=temperature,
        max_retries=2,
    )
    llm = LangChainLLM(llm=langchain_llm)
    Settings.llm = llm
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    PERSIST_DIR = "./storage_sentence_window" if use_sentence_window and SENTENCE_WINDOW_AVAILABLE else "./storage"
    index = None
    bm25_retriever = None
    
    if chatbot_mode in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            if use_sentence_window and SENTENCE_WINDOW_AVAILABLE and not os.path.exists(PERSIST_DIR):
                documents = SimpleDirectoryReader("./data").load_data()
                node_parser = SentenceWindowNodeParser.from_defaults(
                    window_size=sentence_window_size,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text"
                )
                nodes = node_parser.get_nodes_from_documents(documents)
                index = VectorStoreIndex(nodes, storage_context=storage_context)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            else:
                index = load_index_from_storage(storage_context)
        except Exception as e:
            st.warning(f"Local knowledge base not found or error loading: {str(e)}. Web search will be primary source for Agentic RAG.")
            index = None
        
        if chatbot_mode == "Hybrid RAG Chatbot":
            bm25_retriever = initialize_bm25(index)
    
    return index, llm, bm25_retriever

def initialize_bm25(_index):
    try:
        all_nodes = []
        docstore = _index.docstore
        for doc_id in docstore.docs.keys():
            doc = docstore.get_document(doc_id)
            node = TextNode(
                text=doc.text,
                id_=doc_id,
                metadata=getattr(doc, 'metadata', {})
            )
            all_nodes.append(node)
        
        if not all_nodes:
            st.warning("No nodes found for BM25 initialization")
            return None
        
        bm25 = SimpleBM25()
        bm25.fit(all_nodes)
        st.success(f"BM25 initialized with {len(all_nodes)} documents")
        return bm25
    except Exception as e:
        st.error(f"Failed to initialize BM25: {str(e)}")
        return None

def create_agents(llm, index):
    web_search_tool = WebSearchTool()
    agents = [
        InfotainmentAgent("Audio Specialist", "Audio Systems", llm, index, web_search_tool),
        InfotainmentAgent("Display Specialist", "Display & UI", llm, index, web_search_tool),
        InfotainmentAgent("Connectivity Specialist", "Connectivity", llm, index, web_search_tool),
        InfotainmentAgent("Installation Specialist", "Installation", llm, index, web_search_tool),
        WebResearchAgent(llm, web_search_tool)
    ]
    return agents

def agentic_rag_query(query: str, llm, index):
    agents = create_agents(llm, index)
    coordinator = AgentCoordinator(agents, llm)
    
    primary_agent, agent_scores = coordinator.route_query(query)
    
    reasoning = f"🤖 **Agent Routing Decision:**\n"
    reasoning += f"Primary Agent: **{primary_agent.name}** (Confidence: {agent_scores[primary_agent.name]:.2f})\n"
    if enable_web_search:
        reasoning += f"🌐 **Web Search**: Enabled (DuckDuckGo)\n"
    reasoning += "\n"
    
    if enable_multi_agent:
        responses = {}
        source_nodes = []
        web_sources = []
        
        primary_response = primary_agent.process_query(query)
        responses[primary_agent.name] = str(primary_response)
        if hasattr(primary_response, 'source_nodes'):
            source_nodes.extend(primary_response.source_nodes)
        if hasattr(primary_response, 'web_sources'):
            web_sources.extend(primary_response.web_sources)
        
        secondary_candidates = [a for a in agents if a != primary_agent and agent_scores[a.name] > 0.1]
        if secondary_candidates:
            secondary_agent = max(secondary_candidates, key=lambda a: agent_scores[a.name])
            secondary_response = secondary_agent.process_query(query)
            responses[secondary_agent.name] = str(secondary_response)
            if hasattr(secondary_response, 'source_nodes'):
                source_nodes.extend(secondary_response.source_nodes)
            if hasattr(secondary_response, 'web_sources'):
                web_sources.extend(secondary_response.web_sources)
            
            reasoning += f"Secondary Agent: **{secondary_agent.name}** (Confidence: {agent_scores[secondary_agent.name]:.2f})\n\n"
        
        if len(responses) > 1:
            final_response = coordinator.synthesize_response(query, responses)
            reasoning += "🔄 **Multi-Agent Synthesis**: Combining specialist knowledge\n\n"
        else:
            final_response = responses[primary_agent.name]
    else:
        primary_response = primary_agent.process_query(query)
        final_response = str(primary_response)
        source_nodes = primary_response.source_nodes if hasattr(primary_response, 'source_nodes') else []
        web_sources = primary_response.web_sources if hasattr(primary_response, 'web_sources') else []
    
    return final_response, source_nodes, web_sources, reasoning

def get_chat_history_context(max_messages: int = 4) -> str:
    if not use_chat_history or len(st.session_state.messages) < 2:
        return ""
    
    recent_messages = st.session_state.messages[-max_messages:]
    
    history_context = "Recent conversation context:\n"
    for msg in recent_messages:
        if msg["role"] == "user":
            history_context += f"User asked: {msg['content']}\n"
        elif msg["role"] == "assistant":
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            history_context += f"Assistant answered: {content}\n"
    
    return history_context + "\n"

def rewrite_query_with_history(query: str, llm) -> str:
    if not use_query_rewriting:
        return query
    
    chat_context = get_chat_history_context()
    
    query_rewrite_template = PromptTemplate(
        """You are an AI expert specializing in automotive infotainment systems.
        Rewrite the following user query to make it more specific and effective for retrieval 
        from a knowledge base about in-vehicle infotainment systems.
        
        {chat_context}
        Current query: {query}
        
        Consider the conversation history to understand context and intent.
        Provide only the rewritten query that captures the full intent.
        """
    )
    
    response = llm.complete(query_rewrite_template.format(
        chat_context=chat_context,
        query=query
    ))
    return response.text.strip()

def hybrid_retrieve(query: str, vector_retriever, bm25_retriever, top_k: int = 8):
    semantic_nodes = vector_retriever.retrieve(query)
    syntactic_nodes = bm25_retriever.retrieve(query, top_k) if bm25_retriever else []
    hybrid_nodes = {}
    
    if semantic_nodes:
        max_semantic_score = max([node.score for node in semantic_nodes]) if semantic_nodes else 1.0
        for node in semantic_nodes:
            node_id = getattr(node.node, 'id_', str(hash(node.node.text[:100])))
            normalized_score = (node.score / max_semantic_score) * semantic_weight
            if node_id in hybrid_nodes:
                hybrid_nodes[node_id]['score'] += normalized_score
                hybrid_nodes[node_id]['sources'].append('semantic')
            else:
                hybrid_nodes[node_id] = {
                    'node': node,
                    'score': normalized_score,
                    'sources': ['semantic']
                }
    
    if syntactic_nodes:
        max_syntactic_score = max([node.score for node in syntactic_nodes]) if syntactic_nodes else 1.0
        for node in syntactic_nodes:
            node_id = getattr(node.node, 'id_', str(hash(node.node.text[:100])))
            normalized_score = (node.score / max_syntactic_score) * syntactic_weight if max_syntactic_score > 0 else 0
            if node_id in hybrid_nodes:
                hybrid_nodes[node_id]['score'] += normalized_score
                hybrid_nodes[node_id]['sources'].append('syntactic')
            else:
                hybrid_nodes[node_id] = {
                    'node': node,
                    'score': normalized_score,
                    'sources': ['syntactic']
                }
    
    final_nodes = []
    for node_data in hybrid_nodes.values():
        hybrid_node = NodeWithScore(
            node=node_data['node'].node,
            score=node_data['score']
        )
        if hasattr(hybrid_node.node, 'metadata'):
            hybrid_node.node.metadata['retrieval_sources'] = node_data['sources']
        final_nodes.append(hybrid_node)
    
    final_nodes.sort(key=lambda x: x.score, reverse=True)
    return final_nodes[:top_k]

def iterative_refinement(query: str, nodes: List[NodeWithScore], llm) -> str:
    if not use_iterative_refinement or len(nodes) <= 1:
        context = "\n\n".join([node.text for node in nodes[:5]])
        return generate_response_with_context(query, context, llm)
    
    refined_answer = ""
    sorted_nodes = sorted(nodes, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    
    for iteration in range(min(max_refinement_iterations, len(sorted_nodes))):
        current_chunk = sorted_nodes[iteration].text
        refinement_template = PromptTemplate(
            """You are an expert in automotive infotainment systems. 
            
            Original Query: {query}
            Current Answer (if any): {current_answer}
            New Context Chunk: {new_chunk}
            
            Task: {'Generate an initial comprehensive answer' if not refined_answer else 'Refine and enhance the existing answer'} 
            based on the new context chunk. {'Start fresh with this context.' if not refined_answer else 'Integrate new information, correct any inconsistencies, and provide a more complete response.'}
            
            Guidelines:
            - Provide detailed, technical information about infotainment systems
            - Structure the response clearly with relevant sections
            - Focus on accuracy and completeness
            - Minimum 150 words for comprehensive coverage
            
            Enhanced Answer:
            """
        )
        response = llm.complete(refinement_template.format(
            query=query,
            current_answer=refined_answer,
            new_chunk=current_chunk
        ))
        refined_answer = response.text.strip()
        time.sleep(0.1)
    
    return refined_answer

def generate_response_with_context(query: str, context: str, llm) -> str:
    chat_context = get_chat_history_context()
    prompt_template = PromptTemplate(
        """You are an expert assistant specializing in in-vehicle infotainment systems.
        
        {chat_context}
        
        Provide a detailed, accurate, and comprehensive response (at least 150 words) to the following query 
        based on the provided context and conversation history. 
        
        Structure your response clearly and emphasize technical details relevant to in-vehicle infotainment systems.
        If the context doesn't fully answer the query, clearly state what information is missing.

        Query: {query_str}
        Context: {context_str}
        
        Response:
        """
    )
    response = llm.complete(prompt_template.format(
        chat_context=chat_context,
        query_str=query,
        context_str=context
    ))
    return response.text.strip()

def simple_llm_rerank(query: str, nodes: List[NodeWithScore], llm) -> List[NodeWithScore]:
    if not use_reranking or len(nodes) <= 1:
        return nodes
    for node in nodes:
        text_snippet = node.text[:300]
        if any(keyword.lower() in text_snippet.lower() for keyword in query.split()):
            if hasattr(node, 'score'):
                node.score = min(1.0, node.score * 1.2)
    nodes.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    return nodes

def get_enhanced_query_engine(index, llm, bm25_retriever=None):
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
    
    if use_sentence_window and SENTENCE_WINDOW_AVAILABLE:
        postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="window")
        )

    class EnhancedCustomQueryEngine(RetrieverQueryEngine):
        def __init__(self, vector_retriever, bm25_retriever=None, **kwargs):
            super().__init__(retriever=vector_retriever, **kwargs)
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            
        def retrieve(self, query_str):
            enhanced_query = rewrite_query_with_history(query_str, llm)
            if chatbot_mode == "Hybrid RAG Chatbot" and self.bm25_retriever:
                nodes = hybrid_retrieve(enhanced_query, self.vector_retriever, self.bm25_retriever, top_k)
            else:
                nodes = self.vector_retriever.retrieve(enhanced_query)
            nodes = simple_llm_rerank(query_str, nodes, llm)
            return nodes
        
        def query(self, query_str):
            nodes = self.retrieve(query_str)
            if use_iterative_refinement and len(nodes) > 1:
                response_text = iterative_refinement(query_str, nodes, llm)
            else:
                context = "\n\n".join([node.text for node in nodes[:5]])
                response_text = generate_response_with_context(query_str, context, llm)
            
            class EnhancedResponse:
                def __init__(self, text, source_nodes):
                    self.response = text
                    self.source_nodes = source_nodes
                
                def __str__(self):
                    return self.response
            
            return EnhancedResponse(response_text, nodes)

    query_engine = EnhancedCustomQueryEngine(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        node_postprocessors=postprocessors
    )
    return query_engine

def simple_chatbot_query_enhanced(query: str, llm):
    chat_context = get_chat_history_context()
    prompt_template = PromptTemplate(
        """You are an expert assistant specializing in in-vehicle infotainment systems.
        
        {chat_context}
        
        Provide a detailed, accurate, and comprehensive response (at least 150 words) to the following query 
        based on your knowledge and the conversation context.
        
        Structure your response clearly with relevant sections, emphasizing technical details 
        relevant to in-vehicle infotainment systems.

        Query: {query_str}
        Response:
        """
    )
    response = llm.complete(prompt_template.format(
        chat_context=chat_context,
        query_str=query
    ))
    return response.text, []

def main():
    try:
        index, llm, bm25_retriever = load_rag_components()
        
        if chatbot_mode in ["Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
            with st.expander("🚀 Enhanced Features Active"):
                features = []
                if chatbot_mode == "Hybrid RAG Chatbot":
                    features.append("✅ Hybrid Search (Semantic + Syntactic)")
                    features.append(f"✅ Semantic Weight: {semantic_weight:.1f}")
                    features.append(f"✅ Syntactic Weight: {syntactic_weight:.1f}")
                elif chatbot_mode == "Agentic RAG Chatbot":
                    features.append("✅ Specialized Agent Routing")
                    if enable_multi_agent:
                        features.append("✅ Multi-Agent Collaboration")
                    if enable_web_search:
                        features.append("✅ Web Search (DuckDuckGo)")
                    if show_agent_reasoning:
                        features.append("✅ Agent Reasoning Display")
                if use_iterative_refinement:
                    features.append("✅ Iterative Refinement")
                if use_sentence_window and SENTENCE_WINDOW_AVAILABLE:
                    features.append("✅ Sentence Window Retrieval")
                if use_chat_history:
                    features.append("✅ Chat History Context")
                if use_query_rewriting:
                    features.append("✅ Query Rewriting")
                if use_reranking:
                    features.append("✅ Advanced Re-ranking")
                st.markdown('\n'.join(features))
        
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
        
        user_query = st.chat_input("Ask about in-vehicle infotainment systems...")
        
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                spinner_text = f"Processing with {chatbot_mode}..."
                if chatbot_mode == "Agentic RAG Chatbot":
                    spinner_text = "Agents analyzing and searching web..."
                with st.spinner(spinner_text):
                    if chatbot_mode == "Simple Chatbot":
                        response_text, source_nodes = simple_chatbot_query_enhanced(user_query, llm)
                        web_sources = []
                        reasoning_text = ""
                    elif chatbot_mode == "Agentic RAG Chatbot":
                        response_text, source_nodes, web_sources, reasoning_text = agentic_rag_query(user_query, llm, index)
                    else:
                        query_engine = get_enhanced_query_engine(index, llm, bm25_retriever)
                        response = query_engine.query(user_query)
                        response_text = str(response)
                        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                        web_sources = []
                        reasoning_text = ""
                    
                    message_placeholder.markdown(response_text)
                    if reasoning_text and show_agent_reasoning and chatbot_mode == "Agentic RAG Chatbot":
                        with st.expander("🧠 Agent Reasoning"):
                            st.markdown(reasoning_text)
                    if web_sources:
                        with st.expander("🌐 Web Search Results"):
                            for i, source in enumerate(web_sources):
                                st.markdown(f"**Web Source {i+1}:** [{source['title']}]({source['href']})")
                                st.markdown(f"**Content:** {source['body'][:300]}...")
                                st.markdown(f"**Provider:** {source['source']}")
                                st.markdown("---")
                    if source_nodes and chatbot_mode in ["RAG Chatbot", "Advanced RAG Chatbot", "Hybrid RAG Chatbot", "Agentic RAG Chatbot"]:
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