# intelligent_rag_system/src/frontend/app.py

import os
import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional

# --- Configuration (Adjust as needed) ---
# When running with docker-compose, this will be `http://fastapi_app:8000`
# When running FastAPI directly (e.g., uvicorn src.main:app), use `http://localhost:8000`   
FASTAPI_URL = "http://localhost:8000" # Default for local direct run
if st.runtime.exists(): # Adjust for Streamlit running inside Docker Compose
    # Check if running inside a Docker container (implies docker-compose context)
    # This is a heuristic; more robust would be ENV var passed via docker-compose
    # if "FASTAPI_BACKEND_URL" in st.secrets: # Using Streamlit secrets for env var in prod
    #     FASTAPI_URL = st.secrets["FASTAPI_BACKEND_URL"]
    # if "FASTAPI_BACKEND_URL" in st.session_state: # Using session_state for dynamic setting
    #     FASTAPI_URL = st.session_state["FASTAPI_BACKEND_URL"]
    
    # Get from environment variable
    ENV_FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL")
    if ENV_FASTAPI_URL:
        FASTAPI_URL = ENV_FASTAPI_URL
    else: # Fallback for local docker-compose
        FASTAPI_URL = "http://fastapi_app:8000" # Default service name in docker-compose

API_QUERY_ENDPOINT = f"{FASTAPI_URL}/api/query"

# --- Streamlit App UI ---
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="centered")

st.title("Enterprise Knowledge Assistant")
st.markdown("Ask me anything about our company policies, products, employees, or FAQs!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_for_api" not in st.session_state:
    st.session_state.chat_history_for_api = [] # History format for API

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            # Prepare API request payload
            payload = {
                "query": prompt,
                "chat_history": st.session_state.chat_history_for_api # Pass API-formatted history
            }
            headers = {"Content-Type": "application/json"}

            # Make API call
            response = requests.post(API_QUERY_ENDPOINT, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            data = response.json()
            rag_answer = data.get("answer", "No answer found.")
            source_nodes = data.get("source_nodes", [])
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": rag_answer})
            st.session_state.chat_history_for_api = data.get("chat_history", []) # Update API history from backend

            with st.chat_message("assistant"):
                st.markdown(rag_answer)

                if source_nodes:
                    st.subheader("Relevant Sources:")
                    for i, source in enumerate(source_nodes):
                        source_text = source.get("text", "N/A")
                        metadata = source.get("metadata", {})
                        score = source.get("score")

                        # Format source display
                        source_info = []
                        if metadata.get("file_name"):
                            source_info.append(f"**File:** `{metadata['file_name']}`")
                        elif metadata.get("file_path"):
                            source_info.append(f"**Path:** `{metadata['file_path']}`")
                        if metadata.get("page_label"):
                            source_info.append(f"**Page:** `{metadata['page_label']}`")
                        if metadata.get("row_idx") is not None:
                            source_info.append(f"**Row:** `{metadata['row_idx']}`")
                        if metadata.get("file_type"):
                            source_info.append(f"**Type:** `{metadata['file_type'].upper()}`")
                        if score is not None:
                            source_info.append(f"**Score:** `{score:.4f}`")

                        st.expander(f"Source {i+1} ({', '.join(source_info)})").markdown(
                            f"```\n{source_text[:500]}...\n```" # Truncate long source text
                        )

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the FastAPI backend at {FASTAPI_URL}. Please ensure the backend is running.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The backend might be too slow or busy.")
        except requests.exceptions.RequestException as e:
            st.error(f"An API error occurred: {e}. Check the backend logs for details.")
        except json.JSONDecodeError:
            st.error("Received an invalid JSON response from the backend.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Option to clear chat history
st.sidebar.markdown("### Debug Options")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.chat_history_for_api = []
    st.experimental_rerun()

st.sidebar.markdown(f"**FastAPI Backend URL:** `{FASTAPI_URL}`")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Development Notes:**
- Ensure `run_ingestion.py` has been run to populate ChromaDB.
- Ensure `GOOGLE_API_KEY` is set in your `.env` file.
- When running with `docker-compose`, ensure service names (`fastapi_app`) are correctly configured.
""")
