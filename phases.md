# Intelligent Multi-Source RAG System for Enterprise Knowledge Base

This project aims to develop a robust, modular, and scalable Retrieval-Augmented Generation (RAG) system capable of answering complex user queries by synthesizing information from diverse, heterogeneous data sources.

## Project Execution Roadmap: Intelligent Multi-Source RAG System

**Core Principle**: Build iteratively. Get a basic version of each component working, then layer on advanced features.

## Phase 0: Project Initialization & Setup (Days 1-2)

**Goal**: Set up the basic environment and version control.  
**Location in Structure**: Root, .github/workflows/, config/, data/raw/

### Steps:

#### Automate Project Structure Creation:
- Run the create_project.py script you just developed.
- Open the newly created intelligent_rag_system folder in VS Code.

#### Initialize Git Repository:
- Open the integrated terminal in VS Code (Ctrl+ or Cmd+).
```bash
git init
git add .
git commit -m "Initial project structure"
```
- Create a new private repository on GitHub (e.g., intelligent-rag-system).
- Connect your local repo: 
```bash
git remote add origin <your_github_repo_url>
git push -u origin main
```

#### Set Up Python Environment:
- Create a virtual environment: `python -m venv venv`
- Activate it:
  - Windows: `.\venv\Scripts\activate`
  - macOS/Linux: `source venv/bin/activate`
- Add venv/ to your .gitignore.

#### Install Core Dependencies:
Add essential packages to requirements.txt:
- langchain-google-genai
- llama-index
- fastapi
- uvicorn
- streamlit
- spacy
- chromadb (or qdrant-client)
- sentence-transformers
- prefect
- python-dotenv (for managing API keys)
- tiktoken (for token counting with LLMs)
- requests (for HTTP calls)
- pandas (for data handling)
- SQLAlchemy (for SQLite/database interaction)

Install them: `pip install -r requirements.txt`
Download SpaCy model: `python -m spacy download en_core_web_sm`

#### Configure Environment Variables:
Create a .env file in the root directory (remember it's in .gitignore):
```
GOOGLE_API_KEY="your_gemini_api_key_here"
PREFECT_API_URL="https://api.prefect.cloud/api"
PREFECT_API_KEY="your_prefect_cloud_api_key"
```
In config/settings.py, use python-dotenv to load these.

#### Initial Data Loading:
Place some synthetic raw data in data/raw/docs/, data/raw/structured/, data/raw/web_content/ (e.g., a few .txt, .md, .json, .csv files). These will be your starting points.

#### Basic main.yml for CI/CD:
Create a minimal GitHub Actions workflow for linting (e.g., flake8 or black --check) and ensuring Python dependencies are installed.

## Phase 1: Data Ingestion & Indexing Pipeline (Days 3-7)

**Goal**: Build a robust, automated pipeline to get diverse data into your RAG system's indexes.  
**Location in Structure**: src/ingestion/, data/, prefect_flows/

### Steps:

#### Implement Data Loaders (src/ingestion/loaders.py):
- Write functions to load text from .txt, .md files.
- Write functions to load data from .csv and .json into LlamaIndex Document objects (including metadata).
- Write a function to load from a simple SQLite database (e.g., faqs.db with question, answer columns).

#### Define Chunking Logic (src/ingestion/chunking.py):
- Implement LlamaIndex's SentenceSplitter with tuned chunk_size and chunk_overlap.
- Explore and implement basic semantic chunking or parent-document retrieval methods.

#### Initialize Embeddings (src/ingestion/embeddings.py):
- Set up GoogleGenerativeAIEmbeddings from LlamaIndex/LangChain.

#### Vector Store Setup (src/ingestion/vector_store.py):
- Initialize ChromaDB as your persistent vector store, specifying a local directory for its data (e.g., chroma_db_data/). Add chroma_db_data/ to .gitignore.

#### Indexer Logic (src/ingestion/indexer.py):
- Create functions to build and update VectorStoreIndex for unstructured/semi-structured data.
- Create logic to handle structured data, perhaps storing it in an SQLContextContainer if relevant, or simply using it for direct queries later.

#### Develop Prefect Ingestion Flow (prefect_flows/ingestion_flow.py):
Create a Prefect flow that orchestrates the entire data ingestion process:
- Calls loaders.py to extract data.
- Calls chunking.py to chunk documents.
- Calls indexer.py to generate embeddings and store them in ChromaDB.
- Test the flow locally.
- Connect to Prefect Cloud (sign up for free tier, get API key, set PREFECT_API_KEY in .env). Deploy your flow to Prefect Cloud.

#### Data Refresh:
Schedule your Prefect flow to run periodically (e.g., daily) to keep the RAG system updated.

## Phase 2: Core RAG Logic Development (Days 8-15)

**Goal**: Build the intelligent query processing, retrieval, and answer generation modules.  
**Location in Structure**: src/query_understanding/, src/retrieval/, src/llm_generation/

### Steps:

#### Query Understanding (src/query_understanding/):
- **NER (ner_extractor.py)**: Load en_core_web_sm from SpaCy. Create a function to extract entities (e.g., PERSON, ORG, DATE, custom product entities).
- **Query Rewriting (query_rewriter.py)**: Use LangChain's ChatGoogleGenerativeAI (Gemini) with a specific prompt to rephrase or expand the user's initial query for better retrieval.
- **Intent Recognition (intent_recognizer.py)**: (Start simple) Based on keywords or extracted NER, define a simple rule-based system or use a small Gemini call to classify query intent (e.g., "product_info", "HR_policy", "technical_support").

#### Retrieval (src/retrieval/):
- **Keyword Retriever (keyword_retriever.py)**: Implement BM25 (e.g., using rank_bm25 library) over your text chunks.
- **Hybrid Retriever (hybrid_retriever.py)**: Combine the LlamaIndex VectorStoreRetriever and your KeywordRetriever using LangChain's EnsembleRetriever or custom fusion logic.
- **Re-ranker (reranker.py)**: Load a sentence-transformers cross-encoder model. Create a function that takes retrieved documents and the original query, scores them, and returns a re-ranked list.
- **Query Router (query_router.py)**: Based on the output of intent_recognizer.py, intelligently decide which index(es) (vector store, SQL database) and which retrieval strategies to use.

#### LLM Generation (src/llm_generation/):
- **Chat Model Wrapper (chat_model.py)**: Encapsulate ChatGoogleGenerativeAI(model="gemini-1.5-flash") configuration.
- **Prompt Templates (prompt_templates.py)**: Define clear and structured prompts for Gemini:
  - Main RAG prompt (context, question, instructions).
  - Hallucination prevention prompt.
  - Source citation instructions.
- **Answer Synthesizer (answer_synthesizer.py)**: Takes the final re-ranked context, the user query, and the prompt. Calls the chat_model to generate the answer. Include logic for parsing Gemini's output to extract citations.
- **Conversation Manager (conversation_manager.py)**: Implement LangChain's ConversationBufferWindowMemory to maintain chat history.

## Phase 3: API & Frontend Development (Days 16-20)

**Goal**: Expose the RAG system via a robust API and build an interactive user interface.  
**Location in Structure**: src/api/, src/frontend/

### Steps:

#### FastAPI Models (src/api/models.py):
Define Pydantic models for incoming requests (e.g., QueryRequest(query: str, chat_history: List[Dict])) and outgoing responses (QueryResponse(answer: str, sources: List[Dict], confidence_score: Optional[float])).

#### FastAPI Dependencies (src/api/dependencies.py):
Set up dependency injection for your core RAG pipeline instance (e.g., to load it once and reuse across requests).

#### FastAPI Routes (src/api/routes.py):
Create a /query endpoint that:
- Receives QueryRequest.
- Calls the query understanding, retrieval, and LLM generation modules.
- Returns QueryResponse.
- Add a simple /health endpoint.

#### Main FastAPI App (src/main.py):
- Initialize the FastAPI app.
- Include the API routes.
- Add CORS middleware to allow requests from your Streamlit frontend.

#### Streamlit Frontend (src/frontend/app.py):
- Create a simple chat interface using st.chat_input and st.chat_message.
- Use requests to send user queries to your FastAPI backend.
- Display the generated answer, and neatly render the source citations.
- Manage the chat history state within Streamlit.

## Phase 4: MLOps & Observability Integration (Days 21-25)

**Goal**: Containerize the application and set up basic monitoring.  
**Location in Structure**: Dockerfile, docker-compose.yml, monitoring/, src/utils/metrics_utils.py

### Steps:

#### Dockerize FastAPI Application (Dockerfile):
- Create a Dockerfile for your src/main.py FastAPI app.
- Include Python environment setup, dependency installation, and expose the FastAPI port.

#### Docker Compose Setup (docker-compose.yml):
Define services for:
- Your FastAPI app (builds from Dockerfile).
- ChromaDB (if running as a separate service, or ensure its data is mounted).
- Prometheus (using its official image and prometheus.yml).
- Grafana (using its official image and mounting your rag_system_dashboard.json).
- Define necessary volumes and network configurations.

#### Prometheus Metrics Integration (src/utils/metrics_utils.py):
- Use prometheus_client to create custom metrics (e.g., Histogram for latency, Counter for total queries, Gauge for active requests).
- Instrument your FastAPI endpoints and LLM calls to increment/observe these metrics.
- Ensure your FastAPI app exposes a /metrics endpoint for Prometheus to scrape.

#### Prometheus Configuration (monitoring/prometheus.yml):
Configure Prometheus to scrape metrics from your FastAPI service.

#### Grafana Dashboard (monitoring/grafana/dashboards/rag_system_dashboard.json):
- Start Grafana via Docker Compose.
- Import or manually create a basic dashboard to visualize your RAG system's latency, token usage, query count, and retrieval success rates. Export the JSON.

#### Refine GitHub Actions (.github/workflows/main.yml):
- Add steps to build and push your Docker image to a container registry (e.g., Docker Hub) upon successful pushes to main (if you want to demonstrate full CI/CD).
- Ensure tests and linting run before Docker builds.

## Phase 5: Testing, Refinement & Documentation (Days 26-30)

**Goal**: Ensure the system is robust, performant, and well-documented.  
**Location in Structure**: tests/, README.md

### Steps:

#### Unit Tests (tests/unit/):
Write unit tests for chunking.py, retrieval.py (individual retriever components), api/routes.py (endpoint logic), and query_understanding components. Use pytest.

#### Integration Tests (tests/integration/):
Create a test that runs a full RAG query through the FastAPI endpoint, verifying response content and citations. This tests the integrated system.

#### Performance & Cost Optimization:
- Monitor metrics in Grafana. Identify bottlenecks (e.g., slow retrieval, high token usage).
- Adjust chunking strategies, re-ranker threshold, or prompt engineering to optimize.
- Refine Prompts: Continuously iterate on your prompts to Gemini for better answer quality, conciseness, and adherence to instructions.

#### Comprehensive README.md:
Update the README.md with:
- Detailed project overview and problem statement.
- A clear architecture diagram (draw it and embed the image).
- Detailed setup instructions (local, Docker Compose).
- Usage instructions for both API and Streamlit.
- Explanation of advanced features (hybrid search, re-ranking, query understanding).
- Results, performance notes, and examples.
- Future enhancements.

#### Clean Code & Docstrings:
Ensure all functions and classes have clear docstrings and comments.

#### Final Git Commit:
- Ensure all code, configurations, and documentation are committed.
- `git push` to your GitHub repository.