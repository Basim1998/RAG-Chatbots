import streamlit as st

def get_chat_history_context(max_messages: int = 4) -> str:
    if len(st.session_state.get('messages', [])) < 2:
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