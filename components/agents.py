from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from components.web_search import WebSearchTool

class InfotainmentAgent:
    def __init__(self, name: str, expertise: str, llm, index: Optional[VectorStoreIndex], web_search_tool: Optional[WebSearchTool] = None, settings: Dict[str, Any] = None):
        self.name = name
        self.expertise = expertise
        self.llm = llm
        self.index = index
        self.web_search_tool = web_search_tool
        self.settings = settings or {}
        self.query_engine = self._create_query_engine() if index else None
    
    def _create_query_engine(self):
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.settings.get('top_k', 8)
        )
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=self.settings.get('similarity_cutoff', 0.6))]
        if self.settings.get('use_sentence_window') and self.settings.get('sentence_window_available', False):
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
        if not self.web_search_tool or not self.settings.get('enable_web_search', False):
            return []
        domain_query = f"{query} {self.expertise.lower()}"
        return self.web_search_tool.search(domain_query, max_results)
    
    def process_query(self, query: str):
        local_response = None
        web_results = []
        
        if self.query_engine:
            local_response = self.query_engine.query(query)
        
        if self.settings.get('enable_web_search', False):
            web_results = self.search_web(query, self.settings.get('max_web_results', 3))
        
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
    def __init__(self, llm, web_search_tool, settings: Dict[str, Any] = None):
        super().__init__("Web Research Specialist", "Web Research", llm, None, web_search_tool, settings)
    
    def can_handle_query(self, query: str) -> float:
        web_keywords = ["latest", "new", "recent", "update", "market", "trend"]
        query_lower = query.lower()
        matches = sum(1 for keyword in web_keywords if keyword in query_lower)
        return min(matches * 0.5, 1.0)
    
    def process_query(self, query: str):
        web_results = self.search_web(query, self.settings.get('max_web_results', 3))
        
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
    def __init__(self, agents: List[InfotainmentAgent], llm, settings: Dict[str, Any] = None):
        self.agents = agents
        self.llm = llm
        self.settings = settings or {}
    
    def route_query(self, query: str) -> tuple[InfotainmentAgent, Dict[str, float]]:
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

def create_agents(llm, index, settings: Dict[str, Any] = None):
    web_search_tool = WebSearchTool()
    agents = [
        InfotainmentAgent("Audio Specialist", "Audio Systems", llm, index, web_search_tool, settings),
        InfotainmentAgent("Display Specialist", "Display & UI", llm, index, web_search_tool, settings),
        InfotainmentAgent("Connectivity Specialist", "Connectivity", llm, index, web_search_tool, settings),
        InfotainmentAgent("Installation Specialist", "Installation", llm, index, web_search_tool, settings),
        WebResearchAgent(llm, web_search_tool, settings)
    ]
    return agents

def agentic_rag_query(query: str, llm, index, settings: Dict[str, Any]):
    agents = create_agents(llm, index, settings)
    coordinator = AgentCoordinator(agents, llm, settings)
    
    primary_agent, agent_scores = coordinator.route_query(query)
    
    reasoning = f"🤖 **Agent Routing Decision:**\n"
    reasoning += f"Primary Agent: **{primary_agent.name}** (Confidence: {agent_scores[primary_agent.name]:.2f})\n"
    if settings.get('enable_web_search', False):
        reasoning += f"🌐 **Web Search**: Enabled (DuckDuckGo)\n"
    reasoning += "\n"
    
    if settings.get('enable_multi_agent', False):
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
            reasoning += "🔄 **Multi-Agent Synthesis**: Combining specialist knowledge\n"
        else:
            final_response = responses[primary_agent.name]
    else:
        primary_response = primary_agent.process_query(query)
        final_response = str(primary_response)
        source_nodes = primary_response.source_nodes if hasattr(primary_response, 'source_nodes') else []
        web_sources = primary_response.web_sources if hasattr(primary_response, 'web_sources') else []
    
    return final_response, source_nodes, web_sources, reasoning