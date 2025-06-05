# Intelligent Multi-Source RAG System for Enterprise Knowledge Base

## 🎯 Project Overview

This project delivers a robust, modular, and scalable Retrieval-Augmented Generation (RAG) system engineered to answer complex user queries by synthesizing information from diverse enterprise data. By unifying all knowledge sources into a powerful vector database, the system leverages advanced natural language processing (NLP) to provide accurate, contextually relevant, and well-sourced responses.

Our approach emphasizes a lean, LLM-centric architecture, where the Large Language Model (LLM) is responsible for both understanding nuances in queries and intelligently extracting precise information from the retrieved context, regardless of the original data's structure.

### Key Features

- **Unified Multi-Source Data Ingestion**: Seamlessly processes and indexes documents from various formats (Markdown, Plain Text, PDF, HTML, CSV, JSON) into a single vector store.
- **Advanced Query Understanding**: Utilizes Named Entity Recognition (NER) to identify key entities and LLM-driven Query Rewriting to enhance user queries for optimal retrieval, especially in multi-turn conversations.
- **Hybrid Retrieval with Re-ranking**: Combines Vector Similarity Search (for semantic understanding) with Keyword-based Retrieval (BM25) for comprehensive recall. Results are then refined using a Cross-Encoder Re-ranker to prioritize the most relevant information.
- **LLM-Centric Answer Synthesis**: Leverages Google Gemini to synthesize concise, accurate, and contextually rich answers directly from the retrieved text nodes. The LLM is adept at extracting and formatting specific details (e.g., product prices, employee roles) from unstructured text, even if it originated from structured sources.
- **Conversation Management**: Maintains a sliding window of chat history to provide contextual and coherent responses across multi-turn interactions.
- **Real-time Monitoring**: Integrates Prometheus for capturing key application metrics and Grafana for visualizing system performance and health.
- **Containerized Development & Deployment**: Built with FastAPI for a robust API backend and Streamlit for an intuitive web-based chat interface, all designed for consistent environments via Docker.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Ingestion      │    │   Vector Store  │
│                 │───▶│   Pipeline       │───▶│   (ChromaDB)    │
│ • Documents     │    │ • Loaders        │    │                 │
│ • Structured    │    │ • Chunking       │    │                 │
│ • Web Content   │    │ • Embeddings     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Query Engine   │    │   LLM Service   │
│   (Streamlit)   │◀──▶│ • Understanding  │───▶│   (Gemini)      │
│                 │    │ • Retrieval      │    │                 │
└─────────────────┘    │ • Re-ranking     │    └─────────────────┘
                       └──────────────────┘
```

### Architectural Flow:

- **Data Ingestion**: Raw data (documents, CSVs, JSONs, HTML) is loaded, chunked into smaller passages, and transformed into vector embeddings using the embedding model. These embeddings are then stored in ChromaDB, our persistent vector store.

- **User Interaction**: Users interact with the system via the Streamlit Frontend, sending queries to the FastAPI Backend.

- **Query Understanding**: The user's query is first processed by the Query Rewriter (which leverages Gemini and chat history) to enhance its clarity. NER Extractor identifies key entities within the query.

- **Unified Retrieval**: The rewritten query is sent to the Enterprise Retriever, which performs a hybrid search (vector + BM25) across the entire ChromaDB knowledge base to fetch the most semantically and syntactically relevant text passages.

- **Answer Synthesis**: The retrieved text nodes, along with the original user query, are passed to the Answer Synthesizer. Powered by Google Gemini, this component generates a coherent, contextualized answer, carefully extracting specific facts and details directly from the provided text.

- **Conversation Management**: The Conversation Manager updates and maintains the chat history for continuous, context-aware interactions.

## 📁 Project Structure

```
intelligent_rag_system/
├── src/
│   ├── ingestion/           # Data loading and processing
│   │   ├── loaders.py       # File and database loaders
│   │   ├── chunking.py      # Text chunking strategies
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── vector_store.py  # Vector database operations
│   │   └── indexer.py       # Document indexing logic
│   │   └── run_ingestion.py # Ingestion flow
│   ├── query_understanding/ # Query preprocessing
│   │   ├── ner_extractor.py # Named entity recognition
│   │   └── query_rewriter.py# Query enhancement
│   ├── retrieval/           # Information retrieval
│   │   └──retriever.py      # Hybrid retrieval
│   ├── llm_generation/      # Answer generation
│   │   ├── answer_synthesizer.py # Response generation
│   │   └── conversation_manager.py # Conversation management
│   ├── api/                 # FastAPI backend
│   │   ├── models.py        # Pydantic models
│   │   ├── dependencies.py  # Dependency injection
│   │   └── routes.py        # API endpoints
│   ├── frontend/            # Streamlit UI
│   │   └── app.py           # Chat interface
│   └── utils/               # Utilities
│       └── metrics_utils.py # Prometheus metrics
├── data/                    # Data storage
│   ├── raw/                 # Source data
│   └── processed/           # Processed data
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── config/                  # Configuration
│   └── settings.py          # Application settings
├── .env                     # Environment variables
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service deployment
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start  (Local Development)
This section guides you through setting up and running the RAG system on your local machine for development and testing.

### Prerequisites

- Python 3.12+: Recommended to use a virtual environment.
- Docker and Docker Compose: For containerized deployment.
- Google AI API key (for Gemini): For answer generation. (You can get it from [here](https://aistudio.google.com/app/apikey))
- Git: For version control.

### Setup Steps

1. **Clone and Setup Environment**
   ```bash
   git clone <https://github.com/phanhoang1803/rag_system.git>
   cd rag_system
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Configure Environment Variables** \
   Create a .env file in the project root (rag_system/.env) and add your Google AI API key.
   ```bash
   GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
   ```

5. **Prepare Synthetic Raw Data** \
   Project is designed to use synthetic data for demonstration. Ensure you have the following files in their respective data/raw/ subdirectories:

- data/raw/docs/: .md, .txt, .pdf files
- data/raw/structured/: .json, .csv files
- data/raw/web_content/: .html files

6. **Run Data Ingestion and Indexing (Ingestion Flow)** \
   This script will load all your raw data, chunk it, generate embeddings, and store it in the chroma_data/ directory. Run this whenever your raw data changes.
   ```bash
   python src/ingestion/run_ingestion.py
   ```

7. **Start the Application Servers** \
   This will start both the FastAPI backend and the Streamlit frontend.

   - Terminal 1: Start FastAPI Backend
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   - Terminal 2: Start Streamlit frontend
   ```bash
   streamlit run src/frontend/app.py --server.port 8501 --server.enableCORS false
   ```

#### Accessing the Local Application

Once both servers are running:

- Streamlit Chatbot Frontend: Open your web browser and navigate to http://localhost:8501
- FastAPI API Documentation (Swagger UI): Access the interactive API docs at http://localhost:8000/docs
- FastAPI Health Check: Check the API status at http://localhost:8000/api/health

### 🐳 Docker Deployment (Optional)
For consistent and isolated environments, you can containerize and deploy your application using Docker Compose.

1. **Build and Run with Docker Compose** \
   From the project root, execute:
   ```bash
   docker-compose up --build
   ```

This command will build the necessary Docker images, start the FastAPI and Streamlit services.

2. **Access Services in Docker**

- Streamlit Frontend: http://localhost:8501
- FastAPI Backend: http://localhost:8000

## 💡 Usage
Interact with the Enterprise RAG System through the Streamlit web interface or directly via the FastAPI API.

### Streamlit Interface
1. Navigate to http://localhost:8501
2. Type your query into the chat input field
3. The system will generate a response, augmented with relevant source citations from the knowledge base. (For the first time, it will take a while to generate the response)
4. Continue the conversation with follow-up questions; the system maintains context

### Example Queries
- "What are the eligibility criteria for the remote work policy?"
- "Tell me about the features and price of the Cloud Storage Premium product."
- "Who is Alice Smith and what department is she in? What's her role and email?"
- "How do I submit an expense report?"
- "Can you summarize the Q2 business review?"

### 🔧 Advanced Features

- **Named Entity Recognition (NER)**: Extracts key entities (persons, organizations, products, dates) from queries to enhance understanding and guide LLM focus.
- **Query Rewriting**: Dynamically expands or rephrases user queries based on conversation history using a generative LLM, leading to more precise retrieval.
- **Hybrid Retrieval**: Combines the semantic understanding of vector search with the exact keyword matching of BM25 for comprehensive information retrieval.
- **Cross-Encoder Re-ranking**: Employs a pre-trained cross-encoder model to re-score and re-rank initial retrieval results, ensuring that the most relevant documents are presented to the LLM.
- **LLM-driven Fact Extraction**: The generative LLM is prompted to meticulously extract specific details (e.g., names, prices, dates) from the retrieved text, even if the information originated from structured data formats.
- **Sliding Window Conversation Memory**: Maintains a configurable window of recent messages to provide coherent and context-aware responses in multi-turn dialogues.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using LangChain, LlamaIndex, FastAPI, and Streamlit**

---

**Author: Phan Hoang** \
**This README file is supported by AI**