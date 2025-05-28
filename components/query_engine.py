from typing import List
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from components.retrieval import hybrid_retrieve, rewrite_query_with_history
from utils.helpers import get_chat_history_context
import time

def simple_llm_rerank(query: str, nodes: List[NodeWithScore], llm, settings: dict) -> List[NodeWithScore]:
    if not settings.get('use_reranking', False) or len(nodes) <= 1:
        return nodes
    for node in nodes:
        text_snippet = node.text[:300]
        if any(keyword.lower() in text_snippet.lower() for keyword in query.split()):
            if hasattr(node, 'score'):
                node.score = min(1.0, node.score * 1.2)
    nodes.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    return nodes

def iterative_refinement(query: str, nodes: List[NodeWithScore], llm, settings: dict) -> str:
    if not settings.get('use_iterative_refinement', False) or len(nodes) <= 1:
        context = "\n\n".join([node.text for node in nodes[:5]]) if nodes else "No context available."
        return generate_response_with_context(query, context, llm, settings)
    
    refined_answer = ""
    sorted_nodes = sorted(nodes, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    
    for iteration in range(min(settings.get('max_refinement_iterations', 3), len(sorted_nodes))):
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

def generate_response_with_context(query: str, context: str, llm, settings: dict) -> str:
    chat_context = get_chat_history_context(max_messages=4) if settings.get('use_chat_history', True) else ""
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

def simple_chatbot_query_enhanced(query: str, llm, settings: dict):
    chat_context = get_chat_history_context(max_messages=4) if settings.get('use_chat_history', True) else ""
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

def get_enhanced_query_engine(index, llm, bm25_retriever, settings: dict):
    if index is None:
        raise ValueError("Index is None. Cannot create query engine without a valid knowledge base.")
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.get('top_k', 8)
    )
    postprocessors = [SimilarityPostprocessor(similarity_cutoff=settings.get('similarity_cutoff', 0.6))]
    
    if settings.get('use_sentence_window') and settings.get('sentence_window_available', False):
        postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="window")
        )

    class EnhancedCustomQueryEngine(RetrieverQueryEngine):
        def __init__(self, vector_retriever, llm, bm25_retriever=None, settings=None, node_postprocessors=None):
            super().__init__(
                retriever=vector_retriever,
                node_postprocessors=node_postprocessors or [],
            )
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            self.settings = settings or {}
            self.llm = llm
            
        def retrieve(self, query_str):
            enhanced_query = rewrite_query_with_history(query_str, self.llm, self.settings)
            if self.settings.get('chatbot_mode') == "Hybrid RAG Chatbot" and self.bm25_retriever:
                nodes = hybrid_retrieve(enhanced_query, self.vector_retriever, self.bm25_retriever, self.settings)
            else:
                nodes = self.vector_retriever.retrieve(enhanced_query)
            nodes = simple_llm_rerank(query_str, nodes, self.llm, self.settings)
            return nodes
        
        def query(self, query_str):
            try:
                nodes = self.retrieve(query_str)
                if self.settings.get('use_iterative_refinement', False) and nodes:
                    response_text = iterative_refinement(query_str, nodes, self.llm, self.settings)
                else:
                    context = "\n\n".join([node.text for node in nodes[:5]]) if nodes else "No relevant context found."
                    response_text = generate_response_with_context(query_str, context, self.llm, self.settings)
                
                class EnhancedResponse:
                    def __init__(self, text, source_nodes):
                        self.response = text
                        self.source_nodes = source_nodes
                    
                    def __str__(self):
                        return self.response
                
                return EnhancedResponse(response_text, nodes)
            except Exception as e:
                return f"Error processing query: {str(e)}"

    query_engine = EnhancedCustomQueryEngine(
        vector_retriever=vector_retriever,
        llm=llm,
        bm25_retriever=bm25_retriever,
        node_postprocessors=postprocessors,
        settings=settings
    )
    return query_engine