import re
from collections import Counter
import math
import streamlit as st
from typing import List
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.prompts import PromptTemplate
from utils.helpers import get_chat_history_context

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

def initialize_bm25(index):
    try:
        all_nodes = []
        docstore = index.docstore
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

def rewrite_query_with_history(query: str, llm, settings: dict) -> str:
    if not settings.get('use_query_rewriting', False):
        return query
    
    chat_context = get_chat_history_context(max_messages=4)
    
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

def hybrid_retrieve(query: str, vector_retriever, bm25_retriever, settings: dict):
    top_k = settings.get('top_k', 8)
    semantic_weight = settings.get('semantic_weight', 0.7)
    syntactic_weight = settings.get('syntactic_weight', 0.3)
    
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