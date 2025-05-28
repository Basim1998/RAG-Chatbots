from typing import List, Dict
import streamlit as st
from duckduckgo_search import DDGS

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