import os
import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Configuration ---
FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")
API_QUERY_ENDPOINT = f"{FASTAPI_URL}/api/query"

# --- Streamlit App UI Setup ---
st.set_page_config(
    page_title="Enterprise AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'A RAG-powered Enterprise Knowledge Assistant.'
    }
)

# Enhanced Custom CSS with modern design principles
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #64748b;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-primary: #ffffff;
        --background-secondary: #f8fafc;
        --background-tertiary: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-tertiary: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --border-radius: 0.75rem;
        --transition: all 0.2s ease;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-secondary);
        min-height: 100vh;
    }
    
    .block-container {
        max-width: 1400px;
        padding: 2rem;
        margin: 0 auto;
    }
    
    /* Header section */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.125rem;
        font-weight: 400;
    }
    
    /* Chat container - adjusted for more space for chat history */
    .chat-history-container {
        background: var(--background-primary);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        margin-bottom: 1rem;
        min-height: 500px; /* Minimum height for scrollability */
        max-height: calc(100vh - 250px); /* Adjust based on header/input height */
        overflow-y: auto; /* Enable vertical scrolling */
        display: flex; /* Use flexbox for chat message layout */
        flex-direction: column; /* Stack messages vertically */
    }
    
    /* Chat messages */
    .stChatMessage {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: var(--border-radius);
        transition: var(--transition);
        flex-shrink: 0; /* Prevent messages from shrinking */
    }
    
    /* User message styling (right-aligned) */
    .stChatMessage[data-testid="stChatMessage"][data-st-chat-message-role="user"] {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-right: 4px solid var(--primary-color); /* Adjusted border to right */
        align-self: flex-end; /* Align user messages to the right */
        margin-left: auto; /* Push user message to right */
        margin-right: 0; /* Remove right margin */
        max-width: 70%; /* Limit width for readability */
    }
    
    /* Assistant message styling (left-aligned) */
    .stChatMessage[data-testid="stChatMessage"][data-st-chat-message-role="assistant"] {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid var(--success-color);
        align-self: flex-start; /* Align assistant messages to the left */
        margin-right: auto; /* Push assistant message to left */
        margin-left: 0; /* Remove left margin */
        max-width: 70%; /* Limit width for readability */
    }
    
    /* Message content styling */
    .stChatMessage p {
        color: var(--text-primary);
        line-height: 1.6;
        margin: 0;
    }
    
    /* Source cards - now within expanders */
    .source-card {
        background: var(--background-tertiary);
        border: 1px solid var(--border-color);
        border-radius: calc(var(--border-radius) / 2);
        padding: 1rem;
        margin-top: 0.5rem;
        transition: var(--transition);
    }
    
    .source-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* Expander styling for sources */
    .streamlit-expanderHeader {
        background-color: var(--background-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important; /* Slightly more rounded */
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        transition: var(--transition) !important;
        margin-top: 0.75rem; /* Space between message and source expander */
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--background-secondary) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 var(--border-radius) var(--border-radius) !important; /* Matches header radius */
        padding: 1rem !important;
        background-color: var(--background-primary) !important;
    }

    /* Target specific Streamlit elements for better styling */
    div[data-testid="stVerticalBlock"] > div.element-container > div.stButton > button {
        /* Style for buttons in example questions */
        width: 100%;
        margin-bottom: 0.5rem;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: calc(var(--border-radius) / 2);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }
    
    div[data-testid="stVerticalBlock"] > div.element-container > div.stButton > button:hover {
        background: var(--primary-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }

    /* Input container for chat_input */
    /* Adjust this to target the input itself, or the container if it makes sense for layout */
    .st-emotion-cache-1c7y2qn { /* This targets the outer container of the chat_input widget */
        background: var(--background-primary);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        padding: 1rem;
        margin-top: 1rem; /* Add margin to separate from history */
        border: 2px solid transparent;
        transition: var(--transition);
    }
    
    .st-emotion-cache-1c7y2qn:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Buttons in sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: calc(var(--border-radius) / 2);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
        text-transform: none;
        letter-spacing: 0;
        width: 100%; /* Ensure sidebar buttons are full width */
        margin-bottom: 0.5rem; /* Add some space between sidebar buttons */
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--primary-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--background-tertiary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem;
    }
    
    /* Sidebar headers */
    section[data-testid="stSidebar"] h2 {
        color: var(--text-primary);
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Info boxes */
    .info-box {
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: calc(var(--border-radius) / 2);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .info-box h3 {
        color: var(--text-primary);
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        text-align: center;
        color: var(--primary-color);
    }
    
    /* Error/Warning/Success messages */
    .stAlert {
        border-radius: calc(var(--border-radius) / 2);
        border: 1px solid;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stAlert[data-baseweb="notification"][data-kind="error"] {
        background-color: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    
    .stAlert[data-baseweb="notification"][data-kind="warning"] {
        background-color: #fffbeb;
        border-color: #fed7aa;
        color: #92400e;
    }
    
    .stAlert[data-baseweb="notification"][data-kind="success"] {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    
    /* Accessibility improvements */
    *:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Skip to content link */
    .skip-to-content {
        position: absolute;
        top: -40px;
        left: 0;
        background: var(--primary-color);
        color: white;
        padding: 0.5rem 1rem;
        text-decoration: none;
        border-radius: 0 0 var(--border-radius) 0;
        transition: var(--transition);
    }
    
    .skip-to-content:focus {
        top: 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-container {
            padding: 2rem 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        .stChatMessage[data-testid="stChatMessage"][data-st-chat-message-role="user"],
        .stChatMessage[data-testid="stChatMessage"][data-st-chat-message-role="assistant"] {
            max-width: 100%; /* Allow full width on small screens */
            margin-left: 0;
            margin-right: 0;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-primary: #0f172a;
            --background-secondary: #1e293b;
            --background-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --border-color: #475569;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Section with improved visual hierarchy
st.markdown("""
<div class="header-container">
    <h1>🚀 RAG System</h1>
    <p class="header-subtitle">Assistant for knowledge base with RAG</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_welcome = True # Control welcome message visibility

if "chat_history_for_api" not in st.session_state:
    st.session_state.chat_history_for_api = []

# This variable will store the prompt from example buttons
if "example_prompt_to_process" not in st.session_state:
    st.session_state.example_prompt_to_process = ""

# Main chat interface uses two columns
# Swapped col1 and col2 assignment to put the panel on the left
col1, col2 = st.columns([1, 3]) # Changed to [1, 3] to make panel narrower and chat wider

# Function to handle processing of a prompt (either from chat_input or example button)
def process_prompt(prompt_text: str):
    if not prompt_text:
        return

    # Hide welcome message if it was shown and a prompt is submitted
    st.session_state.show_welcome = False
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_text)
    
    # Show assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                # Prepare API request
                payload = {
                    "query": prompt_text,
                    "chat_history": st.session_state.chat_history_for_api
                }
                headers = {"Content-Type": "application/json"}
                
                # Make API call
                response = requests.post(
                    API_QUERY_ENDPOINT, 
                    headers=headers, 
                    json=payload, 
                    timeout=90
                )
                response.raise_for_status()
                
                data = response.json()
                answer = data.get("answer", "I couldn't find relevant information. Please try rephrasing your question.")
                source_nodes = data.get("source_nodes", [])
                
                # Update chat history from backend (critical for conversation flow)
                st.session_state.chat_history_for_api = data.get("chat_history", [])
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Add assistant's message and sources to session state for display on next rerun
                message_data = {"role": "assistant", "content": answer}
                if source_nodes:
                    message_data["sources"] = source_nodes
                st.session_state.messages.append(message_data)
                
                # Display sources in the current message immediately
                if source_nodes:
                    with st.expander(f"📚 View {len(source_nodes)} source(s)", expanded=True):
                        for i, source in enumerate(source_nodes):
                            st.markdown(f"**Source {i+1}**")
                            
                            # Source metadata
                            metadata_items = []
                            if source.get("metadata", {}).get("file_name"):
                                metadata_items.append(f"📄 {source['metadata']['file_name']}")
                            if source.get("metadata", {}).get("page_label"):
                                metadata_items.append(f"📑 Page {source['metadata']['page_label']}")
                            if source.get("score") is not None:
                                metadata_items.append(f"🎯 Score: {source['score']:.2%}")
                            
                            st.caption(" • ".join(metadata_items))
                            
                            # Source text
                            source_text = source.get("text", "Content not available.")
                            display_text = source_text[:500] + "..." if len(source_text) > 500 else source_text
                            st.text_area(label="Content", value=display_text, height=100, disabled=True, 
                                         key=f"source_current_run_{i}_{hash(prompt_text)}") # Unique key for text_area in current run
                            
                            if i < len(source_nodes) - 1:
                                st.divider()
                
            except requests.exceptions.ConnectionError:
                st.error("""
                ❌ **Connection Error**
                
                Could not reach the backend service. Please ensure:
                - The FastAPI backend is running
                - The correct URL is configured
                - Network connectivity is available
                """)
            except requests.exceptions.Timeout:
                st.error("""
                ⏱️ **Request Timeout**
                
                The backend took too long to respond. This might be due to:
                - Large document processing
                - High server load
                - Network latency
                
                Please try again in a moment.
                """)
            except requests.exceptions.HTTPError as e:
                st.error(f"""
                🚫 **API Error**
                
                Status Code: {e.response.status_code}
                
                {e.response.text}
                """)
            except Exception as e:
                st.error(f"""
                💥 **Unexpected Error**
                
                {str(e)}
                
                Please check the server logs for more details.
                """)

# --- Sidebar with improved organization (now in col1 - left) ---
with col1: # All sidebar content within col1 (left column)
    st.markdown("## 🎛️ Control Panel")
    
    # Quick actions section
    st.markdown("### Quick Actions")
    
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_for_api = []
        st.session_state.show_welcome = True
        st.session_state.example_prompt_to_process = "" # Clear pre-filled prompt
        st.rerun() # Rerun to clear chat and show welcome message
    
    chat_export = {
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    st.download_button(
        label="📥 Export Chat",
        data=json.dumps(chat_export, indent=2),
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        key="download_chat_button" # Unique key for the download button
    )
    
    st.divider()
    
    # System status
    st.markdown("### 🔧 System Status")
    
    # Check backend connection - improved robustness
    backend_status_message = "🔴 Backend Offline"
    backend_status_icon = "❌"
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=3) # Increased timeout slightly
        if response.status_code == 200:
            status_data = response.json()
            if status_data.get('status') == 'ok': # Check for specific 'ok' status from JSON
                backend_status_message = "🟢 Backend Connected"
                backend_status_icon = "✅"
            else:
                backend_status_message = "🟠 Backend Responding (Status not 'ok')"
                backend_status_icon = "⚠️"
        else:
            backend_status_message = f"🟠 Backend Error ({response.status_code})"
            backend_status_icon = "⚠️"
    except requests.exceptions.ConnectionError:
        backend_status_message = "🔴 Backend Offline (Connection refused)"
        backend_status_icon = "❌"
    except requests.exceptions.Timeout:
        backend_status_message = "🟡 Backend Timeout"
        backend_status_icon = "⚠️"
    except json.JSONDecodeError:
        backend_status_message = "🟡 Backend Responded (Invalid JSON)"
        backend_status_icon = "⚠️"
    except Exception as e:
        backend_status_message = f"🔴 Backend Error ({type(e).__name__})"
        backend_status_icon = "❌"
    
    if backend_status_icon == "✅":
        st.success(f"{backend_status_icon} {backend_status_message}")
    elif backend_status_icon == "⚠️":
        st.warning(f"{backend_status_icon} {backend_status_message}")
    else:
        st.error(f"{backend_status_icon} {backend_status_message}")

    # Display endpoint info
    with st.expander("🔌 Connection Details", expanded=False):
        st.code(f"Endpoint: {FASTAPI_URL}", language="text")
        st.caption("Ensure the FastAPI backend is running and accessible.")
    
    st.divider()
    
    # Help section
    st.markdown("### 📚 Help & Resources")
    
    with st.expander("🤔 How to use", expanded=False):
        st.markdown("""
        1. **Ask natural questions** - Type your query in plain English.
        2. **Be specific** - Include relevant details for better results.
        3. **Check sources** - Review the source documents for context.
        4. **Iterate** - Refine your questions based on responses.
        """)
    
    with st.expander("🏗️ System Architecture", expanded=False):
        st.markdown("""
        - **Frontend**: Streamlit UI
        - **Backend**: FastAPI + LlamaIndex
        - **Vector DB**: ChromaDB
        - **LLM**: Google Gemini
        - **Embeddings**: Gemini Embeddings
        """)
    
    st.divider()
    
    # About section
    st.markdown("### ℹ️ About")
    st.caption("""
    RAG System v1.0
    
    Built with ❤️ using modern AI technologies
    
    © 2025 Phan Hoang
    """)

# --- Chat Interface (now in col2 - right) ---
with col2: # All chat content within col2 (right column)
    # --- Welcome Message and Example Questions (Progressive Disclosure) ---
    if st.session_state.show_welcome:
        welcome_container = st.container()
        with welcome_container:
            st.info("""
            👋 **Welcome to RAG System!**
            
            There are some questions you can ask:
            • What are the eligibility criteria for the remote work policy?
            • Tell me about the features and price of the Cloud Storage Premium product.
            • Who is Alice Smith and what department is she in? What's her role and email?
            • Can you summarize the Q2 business review?
            
            Just type your question below to get started!
            """)
            
            st.markdown("### 💡 Try asking:")
            example_cols = st.columns(2)
            
            # Callback for example buttons
            def handle_example_click(question):
                st.session_state.example_prompt_to_process = question
                st.session_state.show_welcome = False # Hide welcome after first interaction
                # No st.rerun() here. We'll process it below the chat_input.
            
            with example_cols[0]:
                st.button("📋 Remote work policy", key="ex1", use_container_width=True,
                          on_click=handle_example_click, args=("What are the eligibility criteria for the remote work policy?",))
                st.button("💰 Product pricing", key="ex2", use_container_width=True,
                          on_click=handle_example_click, args=("Tell me about the features and price of the Cloud Storage Premium product.",))
                    
            with example_cols[1]:
                st.button("👥 Find employee", key="ex3", use_container_width=True,
                          on_click=handle_example_click, args=("Who is Alice Smith and what department is she in?",))
                st.button("📊 Business reports", key="ex4", use_container_width=True,
                          on_click=handle_example_click, args=("Can you summarize the Q2 business review?",))
    
    # --- Chat History Display ---
    # Moved chat history into a scrollable container
    with st.container(height=500): # Fixed height for scrollability
        if st.session_state.messages:
            for message in st.session_state.messages:
                # Use a specific data-testid for custom CSS targeting based on role
                with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                    st.markdown(message["content"])
                    
                    # Display sources if available and part of the message
                    if "sources" in message and message["sources"]:
                        with st.expander(f"📚 View {len(message['sources'])} source(s)", expanded=False):
                            for i, source in enumerate(message["sources"]):
                                # Consistent formatting for source metadata
                                st.markdown(f"**Source {i+1}**")
                                
                                metadata_items = []
                                if source.get("metadata", {}).get("file_name"):
                                    metadata_items.append(f"📄 {source['metadata']['file_name']}")
                                if source.get("metadata", {}).get("page_label"):
                                    metadata_items.append(f"📑 Page {source['metadata']['page_label']}")
                                if source.get("score") is not None:
                                    metadata_items.append(f"🎯 Score: {source['score']:.2%}")
                                
                                st.caption(" • ".join(metadata_items))
                                
                                # Source text with proper formatting and truncation
                                source_text = source.get("text", "Content not available.")
                                display_text = source_text[:500] + "..." if len(source_text) > 500 else source_text
                                st.text_area(label="Content", value=display_text, height=100, disabled=True, 
                                             key=f"source_content_{message['role']}_{hash(message['content'])}_{i}")
                                # Unique key for text_area
                                
                                if i < len(message["sources"]) - 1:
                                    st.divider()

    # --- Chat Input (moved inside col2 to be with chat history) ---
    # Check if an example prompt was set by a button click
    if st.session_state.example_prompt_to_process:
        # Process the example prompt immediately
        process_prompt(st.session_state.example_prompt_to_process)
        # Clear the example prompt so it's not processed again on subsequent reruns
        st.session_state.example_prompt_to_process = ""
        st.rerun() # Rerun to update the chat history and clear the (now processed) example prompt

    # This captures user input from the chat_input widget
    if user_input_prompt := st.chat_input("Ask me anything about our company...", key="chat_input"):
        process_prompt(user_input_prompt)
        st.rerun()
        # No need for st.rerun() here as the chat_input already triggers a rerun on submit,
        # and the process_prompt function adds the message and calls the API.

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-tertiary); font-size: 0.875rem;">
    <p>Need help? Contact phanhoang1803@gmail.com | <a href="#" style="color: var(--primary-color);">Documentation</a> | <a href="#" style="color: var(--primary-color);">Privacy Policy</a></p>
</div>
""", unsafe_allow_html=True)