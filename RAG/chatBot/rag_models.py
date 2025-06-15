import os
import time
import logging
from datetime import datetime
import json
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
import requests
import openai
import threading
from supabase import create_client, Client

# Add Together AI integration
try:
    from together import Together
    TOGETHER_AVAILABLE = True
    print("TogetherAI library successfully imported!")
except ImportError as e:
    TOGETHER_AVAILABLE = False
    print(f"TogetherAI library not available: {e}. Will fall back to OpenAI.")

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"

# Try to get Supabase keys from various sources
supabase_key = None

# 1. Try environment variables
supabase_key = os.getenv('SUPABASE_ANON_KEY')

# 2. Try Streamlit secrets if available
if not supabase_key:
    try:
        if 'SUPABASE_ANON_KEY' in st.secrets:
            supabase_key = st.secrets['SUPABASE_ANON_KEY']
    except Exception as e:
        print(f"Could not access Streamlit secrets: {e}")

# 3. Fallback to hardcoded key if needed
if not supabase_key:
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc5NjI1MTQsImV4cCI6MjA1MzUzODUxNH0.N0SLqiMO6KxAlf_hyNTu1W1RZ8MfltuXwtdc1o-7eAs"

supabase: Client = None
try:
    supabase = create_client(SUPABASE_URL, supabase_key)
    st.session_state.supabase_connected = True
    print("Supabase connection successful in rag_models.py!")
except Exception as e:
    st.session_state.supabase_connected = False
    print(f"Failed to connect to Supabase in rag_models.py: {e}")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom logger handler to also log to Supabase
class SupabaseHandler(logging.Handler): 
    def emit(self, record):
        if not hasattr(st.session_state, 'supabase_connected') or not st.session_state.supabase_connected:
            return
        
        try:
            # Check if table exists
            try:
                supabase.table("system_logs").select("*").limit(1).execute()
                table_exists = True
            except Exception:
                # Table doesn't exist - silently fail
                print("system_logs table doesn't exist in Supabase. Skipping remote logging.")
                return
            
            if table_exists:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "message": self.format(record),
                    "logger": record.name,
                    "pathname": record.pathname,
                    "lineno": record.lineno
                }
                
                # Insert into Supabase
                supabase.table("system_logs").insert(log_entry).execute()
        except Exception as e:
            # Don't use logger here to avoid infinite recursion
            print(f"Failed to log to Supabase: {e}")

# Add Supabase handler if connected
if supabase and st.session_state.get('supabase_connected', False):
    supabase_handler = SupabaseHandler()
    supabase_handler.setLevel(logging.INFO)
    logger.addHandler(supabase_handler)

# Try to get API keys from environment variables first
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')  # Add Together API key

# If not found in environment, try to get from Streamlit secrets
try:
    if not openai_api_key and 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
    if not langchain_api_key and 'LANGCHAIN_API_KEY' in st.secrets:
        langchain_api_key = st.secrets['LANGCHAIN_API_KEY']
    if not together_api_key and 'TOGETHER_API_KEY' in st.secrets:
        together_api_key = st.secrets['TOGETHER_API_KEY']
except Exception as e:
    logger.warning(f"Could not access Streamlit secrets: {e}")

# Set the API keys if found
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
if together_api_key:
    os.environ['TOGETHER_API_KEY'] = together_api_key

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize cache
set_llm_cache(InMemoryCache())

# Define the NASA documents prompt template with source citation
nasa_prompt = ChatPromptTemplate.from_template("""
You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

Context: {context}

Question: {input}

Format:
1. Brief Answer
2. Key Points
3. Relevant Mission Details (if applicable)

DO NOT include a Sources section in your response. The system will add this automatically based on the actual documents used.
""")

# Create a custom langchain LLM class for TogetherAI
# Going for unstructured approach of having several classes in a single file, not the best practice, but had a lot of issues/bugs when running, so having everything in one file made it easier to track issues.
# Move the classes into own files later on.
class TogetherLLM(LLM):
    """Custom LLM class for TogetherAI that inherits from LangChain's LLM base class."""
    
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    temperature: float = 0.1
    max_tokens: int = 1024
    together_client: Any = None
    
    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.1, max_tokens=1024):
        """Initialize TogetherLLM."""
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.together_client = Together()
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together_ai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process the prompt using TogetherAI."""
        try:
            response = self.together_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error using TogetherAI: {e}")
            return f"Error: Could not generate response with TogetherAI ({str(e)})"

class RAGModel:
    def __init__(self, chunks_dir=["../../reprocessed_section_chunks", 
                                  "../../reprocessed_section_chunks_2", 
                                  "../../reprocessed_section_chunks_3"],
                 lessons_learned_path="../../RAG/NASA_Lessons_Learned/nasa_lessons_learned_centers_1.csv",
                 index_dir="vector_indices"):
        # Get the absolute path of the current file
        current_dir = Path(__file__).parent.absolute()
        
        # For vector indices, use a path relative to the current file
        self.index_dir = current_dir / index_dir
        
        # For chunks and lessons learned, use paths relative to the project root
        # First, find the project root (where the RAG directory is)
        project_root = current_dir.parent.parent  # Go up two levels from chatBot to RAG to project root
        
        # Now construct the absolute paths
        self.chunks_dirs = [project_root / dir_path for dir_path in chunks_dir]
        self.lessons_learned_path = project_root / lessons_learned_path

        self.db = None
        self.retriever = None
        self.llm = None
        self.together_llm = None  # Add TogetherAI LLM
        self.model_loading = False
        self.model_loaded = False
        self.together_available = TOGETHER_AVAILABLE and together_api_key is not None
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        self.embed = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Start model loading in background
        self._start_background_model_loading()
        
        logger.info(f"Looking for chunks in: {self.chunks_dirs}")
        
        # Load or create vector store
        self._load_or_create_vector_store()
    
    def _start_background_model_loading(self):
        """Start loading the Mistral model in a background thread"""
        def load_model():
            logger.info("Starting background model loading...")
            self.model_loading = True
            try:
                # MODIFIED: Try TogetherAI first if available
                if self.together_available:
                    logger.info("Trying TogetherAI first...")
                    try:
                        self.together_llm = TogetherLLM()
                        # Test with a simple call
                        test_response = self.together_llm.invoke("Hello")
                        if test_response and not test_response.startswith("Error:"):
                            self.model_loaded = True
                            logger.info("Using TogetherAI Llama-3.3-70B model for responses")
                            return  # Successfully loaded TogetherAI, no need to try Ollama
                        else:
                            logger.warning(f"TogetherAI test call failed with response: {test_response}")
                            raise Exception("TogetherAI test failed")
                    except Exception as e:
                        logger.warning(f"TogetherAI not available: {str(e)}")
                        self.together_llm = None  # Reset to None in case it was partially initialized
                else:
                    logger.warning("TogetherAI not available: library not imported or API key missing")
                
                # WE ARE NOT LONGER USING THE wizardlm2 model, but for it is kept as an possibility when testing locally in the future.
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        self.llm = OllamaLLM(
                            model="wizardlm2",
                            temperature=0.1,
                            num_ctx=512,
                            request_timeout=60.0,
                            num_predict=256,
                            num_thread=4,
                            stop=["4. Sources"]
                        )
                        # Make a dummy call to ensure model is loaded
                        self.llm.invoke("Hello")
                        self.model_loaded = True
                        logger.info("Using Ollama model (wizardlm2) for responses as fallback")
                    else:
                        logger.warning(f"Ollama API returned status code: {response.status_code}")
                        raise Exception("Ollama not available")
                except Exception as ollama_error:
                    logger.warning(f"Ollama not available: {str(ollama_error)}")
                    raise Exception(f"Neither TogetherAI nor Ollama are available. TogetherAI error: {str(e) if 'e' in locals() else 'Not attempted'}, Ollama error: {str(ollama_error)}")
            except Exception as e:
                logger.error(f"Error loading local models: {str(e)}")
                logger.warning("Falling back to OpenAI")
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1024
                )
                self.model_loaded = True
                logger.info("Using OpenAI model (gpt-4o-mini) for responses")
            finally:
                self.model_loading = False
        
        # Start the loading thread
        thread = threading.Thread(target=load_model)
        thread.daemon = True  # Thread will be killed when main program exits
        thread.start()
    
    def _load_chunks(self):
        """Load document chunks from JSON files and CSV files"""
        logger.info(f"Loading document chunks from multiple sources...")
        chunks = []
        
        # Process JSON files from all directories
        for chunks_dir in self.chunks_dirs:
            logger.info(f"Processing directory: {chunks_dir}")
            # Get all JSON files in the chunks directory
            json_files = list(chunks_dir.glob("**/*.json*"))
            
            if not json_files:
                logger.warning(f"No JSON files found in {chunks_dir}")
                # List directory contents to help debug
                try:
                    logger.info(f"Directory contents: {list(chunks_dir.iterdir())}")
                except Exception as e:
                    logger.error(f"Could not list directory contents: {e}")
                continue
                
            logger.info(f"Found {len(json_files)} JSON files in {chunks_dir}")
            
            # Process each JSON file
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Handle both single objects and arrays
                        if isinstance(data, list):
                            items = data
                        else:
                            items = [data]
                        
                        for item in items:
                            if 'content' in item and 'title' in item:
                                # Extract metadata properly
                                metadata_dict = {
                                    'title': item['title'],
                                    'chunk_id': item.get('chunk_id', ''),
                                    'page_number': item.get('page_number', ''),
                                    'section_level': item.get('section_level', ''),
                                    'file_name': item.get('metadata', {}).get('file_name', ''),
                                    'source_type': 'pdf'
                                }
                                
                                # Add download_url directly to the metadata
                                if 'metadata' in item and 'download_url' in item['metadata']:
                                    metadata_dict['download_url'] = item['metadata']['download_url']
                                
                                # Create document with content and metadata
                                doc = Document(
                                    page_content=item['content'],
                                    metadata=metadata_dict
                                )
                                chunks.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        # Process NASA Lessons Learned CSV file
        if self.lessons_learned_path.exists():
            logger.info(f"Processing NASA Lessons Learned from {self.lessons_learned_path}")
            try:
                import pandas as pd
                df = pd.read_csv(self.lessons_learned_path)
                
                # Process each row in the CSV
                for _, row in df.iterrows():
                    # Skip header row if it got included
                    if row.get('url') == 'url':
                        continue
                        
                    # Combine relevant fields into content
                    content_parts = []
                    
                    if not pd.isna(row.get('subject')):
                        content_parts.append(f"Subject: {row['subject']}")
                    
                    if not pd.isna(row.get('abstract')) and row['abstract'] != 'None':
                        content_parts.append(f"Abstract: {row['abstract']}")
                    
                    if not pd.isna(row.get('driving_event')) and row['driving_event'] != 'None':
                        content_parts.append(f"Driving Event: {row['driving_event']}")
                    
                    if not pd.isna(row.get('lessons_learned')) and row['lessons_learned'] != 'None':
                        content_parts.append(f"Lessons Learned: {row['lessons_learned']}")
                    
                    if not pd.isna(row.get('recommendations')) and row['recommendations'] != 'None':
                        content_parts.append(f"Recommendations: {row['recommendations']}")

                    if not pd.isna(row.get('evidence')) and row['evidence'] != 'None':
                        content_parts.append(f"Evidence: {row['evidence']}")

                    if not pd.isna(row.get('program_relation')) and row['program_relation'] != 'None':
                        content_parts.append(f"Program Relation: {row['program_relation']}")

                    if not pd.isna(row.get('program_phase')) and row['program_phase'] != 'None':
                        content_parts.append(f"Program Phase: {row['program_phase']}")

                    if not pd.isna(row.get('mission_directorate')) and row['mission_directorate'] != 'None':
                        content_parts.append(f"Mission Directorate: {row['mission_directorate']}")

                    if not pd.isna(row.get('topics')) and row['topics'] != 'None':
                        content_parts.append(f"Topics: {row['topics']}")
                    
                    content = "\n\n".join(content_parts)
                    
                    # Create metadata
                    metadata_dict = {
                        'title': row.get('subject', 'NASA Lesson Learned'),
                        'url': row.get('url', ''),
                        'source_type': 'lessons_learned',
                        'mission_directorate': row.get('mission_directorate', ''),
                        'program_phase': row.get('program_phase', ''),
                        'topics': row.get('topics', '')
                    }
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata=metadata_dict
                    )
                    chunks.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing NASA Lessons Learned CSV: {e}")
        
        logger.info(f"Loaded {len(chunks)} total document chunks")
        return chunks
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one"""
        index_path = self.index_dir / "nasa_docs_index"
        
        if index_path.exists():
            logger.info("Loading existing vector store...")
            try:
                self.db = FAISS.load_local(
                    str(index_path),
                    self.embed,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                logger.info("Creating new vector store...")
                self._create_vector_store()
        else:
            logger.info("Creating new vector store...")
            self._create_vector_store()
        print("7")
        # Create retriever
        if self.db:
            self.retriever = self.db.as_retriever(
                search_kwargs={
                    "k": 5, # Experimenting with this
                    "score_threshold": 0.7 # this as well
                }
            )
        print("8")
    def _create_vector_store(self):
        """Create a new vector store from document chunks"""
        chunks = self._load_chunks()
        
        if not chunks:
            logger.warning("No chunks to create vector store")
            return
        
        # Create vector store with batching
        logger.info("Creating vector store...")
        BATCH_SIZE = 32
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(total_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(chunks))
            batch = chunks[start_idx:end_idx]
            
            batch_start_time = time.time()
            
            if i == 0:
                self.db = FAISS.from_documents(batch, self.embed)
                print("9")
            else:
                self.db.add_documents(batch)
                print("10")
            batch_duration = time.time() - batch_start_time
            docs_per_second = len(batch) / batch_duration
            logger.info(f"Batch {i+1}/{total_batches} completed in {batch_duration:.2f} seconds ({docs_per_second:.2f} docs/second)")
        
        # Save the vector store
        if self.db:
            self.db.save_local(str(self.index_dir / "nasa_docs_index"))
            logger.info("Vector store created and saved successfully")
    
    def query(self, question, model_name="openai"):
        """Query the RAG model with a question"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "Error: Retriever not initialized. Please try again later."
        
        logger.info(f"Processing question with {model_name}: {question}")
        query_start_time = time.time()
        
        try:
            # Retrieve documents first (for both models)
            if model_name.lower() == "llama":
                # Optimize for Llama
                self.retriever.search_kwargs["k"] = 4  # Reduce number of documents
                self.retriever.search_kwargs["score_threshold"] = 0.8  # Higher threshold for better quality
            else:
                # More documents for OpenAI
                self.retriever.search_kwargs["k"] = 4
            
            # Get documents
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Print document metadata for debugging
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
            
            # Initialize the appropriate LLM based on model_name
            if model_name.lower() == "llama":
                # CHANGED: Always prioritize TogetherAI when model is "llama"
                # First, try to initialize TogetherAI if not already done
                if self.together_available and not self.together_llm:
                    try:
                        self.together_llm = TogetherLLM()
                        # Test with a simple call
                        test_response = self.together_llm.invoke("Hello")
                        if test_response.startswith("Error:"):
                            self.together_llm = None
                            logger.warning("TogetherAI test failed, will use alternative model")
                        else:
                            logger.info("TogetherAI Llama-3.3-70B model initialized on demand")
                    except Exception as e:
                        logger.warning(f"Error initializing TogetherAI: {e}")
                        self.together_llm = None
                
                # Now check if TogetherAI is available to use
                if self.together_available and self.together_llm is not None:
                    # Create a more concise prompt for faster processing
                    llama_prompt = ChatPromptTemplate.from_template("""
                        You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

                        Context: {context}

                        Question: {input}

                        Format:
                        1. Brief Answer
                        2. Key Points
                        3. Relevant Mission Details (if applicable)

                        DO NOT include a Sources section in your response. The system will add this automatically based on the actual documents used.
                    """)
                    
                    # Use TogetherAI LLM
                    combine_docs_chain = create_stuff_documents_chain(self.together_llm, llama_prompt)
                    logger.info("Using TogetherAI Llama-3.3-70B model for this query")
                    
                # Fall back to Ollama if it's available
                elif self.llm is not None:
                    # Create a more concise prompt for faster processing
                    llama_prompt = ChatPromptTemplate.from_template("""
                        You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

                        Context: {context}

                        Question: {input}

                        Format:
                        1. Brief Answer
                        2. Key Points
                        3. Relevant Mission Details (if applicable)

                        DO NOT include a Sources section in your response. The system will add this automatically based on the actual documents used.
                    """)
                    
                    # Use the concise prompt for faster processing
                    combine_docs_chain = create_stuff_documents_chain(self.llm, llama_prompt)
                    logger.info("Using Ollama model (wizardlm2) for this query")
                
                # If neither is available, handle the loading state
                else:
                    if self.model_loading:
                        return "The model is still loading. Please try again in a few moments."
                    else:
                        # Start loading the model if it hasn't been loaded yet
                        self._start_background_model_loading()
                        return "The model is being loaded. Please try again in a few moments."
            
            else:  # Default to OpenAI change this latter, we don't actually want to fallback, give error instead. 
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Use the full NASA prompt for OpenAI
                combine_docs_chain = create_stuff_documents_chain(llm, nasa_prompt)
                logger.info("Using OpenAI model (gpt-40-mini) for this query")
            
            # Create retrieval chain with optimized parameters
            retrieval_chain = create_retrieval_chain(
                self.retriever,
                combine_docs_chain
            )
            
            # Execute the query with timeout
            try:
                result = retrieval_chain.invoke(
                    {'input': question},
                    config={"timeout": 60}  # Add timeout to prevent hanging
                )
                answer = result["answer"]
            except Exception as e:
                logger.error(f"Query timeout or error: {e}")
                return "The query took too long to process. Please try again or use a different model."
            
            # Add source information
            sources = []
            for doc in retrieved_docs:
                source_type = doc.metadata.get("source_type", "unknown")
                
                if source_type == "pdf":
                    file_name = doc.metadata.get("file_name", "Unknown")
                    
                    # Extract download_url directly from metadata
                    download_url = doc.metadata.get("download_url", "No URL available")
                    
                    # If not found, try to find it in the JSON file
                    if download_url == "No URL available":
                        try:
                            chunk_id = doc.metadata.get("chunk_id", "")
                            if chunk_id:
                                # Extract the PDF name from chunk_id
                                pdf_name = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                                # Look for the JSON file with this PDF name
                                for chunks_dir in self.chunks_dirs:
                                    for json_file in chunks_dir.glob("**/*.json*"):
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                            if isinstance(data, list):
                                                for item in data:
                                                    if item.get("chunk_id", "").startswith(pdf_name):
                                                        if "metadata" in item and "download_url" in item["metadata"]:
                                                            download_url = item["metadata"]["download_url"]
                                                            logger.info(f"Found download URL for {pdf_name}: {download_url}")
                                                            break
                        except Exception as e:
                            logger.error(f"Error finding download URL: {e}")
                    
                    sources.append(f"- {file_name}: {download_url}")
                
                elif source_type == "lessons_learned":
                    url = doc.metadata.get("url", "No URL available")
                    title = doc.metadata.get("title", "NASA Lesson Learned")
                    sources.append(f"- NASA Lesson Learned - {title}: {url}")
            
            # Always add sources section
            answer += "\n\n4. Sources:\n" + "\n".join(sources)
            
            query_duration = time.time() - query_start_time
            logger.info(f"Query completed in {query_duration:.2f} seconds")
            
            # Log the query to Supabase if connected
            if st.session_state.get('supabase_connected', False) and supabase:
                try:
                    # Check if table exists
                    try:
                        supabase.table("rag_queries").select("*").limit(1).execute()
                        table_exists = True
                    except Exception:
                        print("rag_queries table doesn't exist in Supabase. Skipping remote logging.")
                        table_exists = False
                    
                    if table_exists:
                        # Get username from session state if available
                        username = st.session_state.get('username', 'anonymous')
                        # Get current task ID from session state if available
                        task_id = st.session_state.get('current_task_id', None)
                        
                        query_log = {
                            "timestamp": datetime.now().isoformat(),
                            "username": username,
                            "task_id": task_id,
                            "question": question,
                            "model": model_name,
                            "processing_time": query_duration,
                            "num_docs_retrieved": len(retrieved_docs),
                            "answer_length": len(answer) if 'answer' in locals() else 0
                        }
                        supabase.table("rag_queries").insert(query_log).execute()
                except Exception as e:
                    logger.error(f"Failed to log query to Supabase: {e}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def get_document_by_id(self, doc_id):
        logger.info(f"Looking for document with ID: {doc_id}")
        
        # First check if it's a lessons learned URL
        if doc_id.startswith("https://"):
            if self.lessons_learned_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(self.lessons_learned_path)
                    lesson = df[df['url'] == doc_id].iloc[0]
                    
                    # Combine relevant fields into content
                    content_parts = []
                    if not pd.isna(lesson.get('subject')):
                        content_parts.append(f"Subject: {lesson['subject']}")
                    if not pd.isna(lesson.get('abstract')) and lesson['abstract'] != 'None':
                        content_parts.append(f"Abstract: {lesson['abstract']}")
                    if not pd.isna(lesson.get('driving_event')) and lesson['driving_event'] != 'None':
                        content_parts.append(f"Driving Event: {lesson['driving_event']}")
                    if not pd.isna(lesson.get('lessons_learned')) and lesson['lessons_learned'] != 'None':
                        content_parts.append(f"Lessons Learned: {lesson['lessons_learned']}")
                    if not pd.isna(lesson.get('recommendations')) and lesson['recommendations'] != 'None':
                        content_parts.append(f"Recommendations: {lesson['recommendations']}")
                    if not pd.isna(lesson.get('evidence')) and lesson['evidence'] != 'None':
                        content_parts.append(f"Evidence: {lesson['evidence']}")
                    if not pd.isna(lesson.get('program_relation')) and lesson['program_relation'] != 'None':
                        content_parts.append(f"Program Relation: {lesson['program_relation']}")
                    if not pd.isna(lesson.get('program_phase')) and lesson['program_phase'] != 'None':
                        content_parts.append(f"Program Phase: {lesson['program_phase']}")
                    if not pd.isna(lesson.get('mission_directorate')) and lesson['mission_directorate'] != 'None':
                        content_parts.append(f"Mission Directorate: {lesson['mission_directorate']}")
                    if not pd.isna(lesson.get('topics')) and lesson['topics'] != 'None':
                        content_parts.append(f"Topics: {lesson['topics']}")
                    
                    content = "\n\n".join(content_parts)
                    
                    # Create metadata
                    metadata_dict = {
                        'title': lesson.get('subject', 'NASA Lesson Learned'),
                        'url': doc_id,
                        'source_type': 'lessons_learned',
                        'mission_directorate': lesson.get('mission_directorate', ''),
                        'program_phase': lesson.get('program_phase', ''),
                        'topics': lesson.get('topics', '')
                    }
                    
                    return Document(
                        page_content=content,
                        metadata=metadata_dict
                    )
                except Exception as e:
                    logger.error(f"Error retrieving lesson learned document: {e}")
                    return None
        
        # If not a URL, treat as a PDF chunk ID
        for chunks_dir in self.chunks_dirs:
            try:
                # The chunk ID format is typically "filename_pagenumber"
                # Extract the filename part
                file_prefix = doc_id.split('_')[0] if '_' in doc_id else doc_id
                
                # Search through all JSON files in the directory
                for json_file in chunks_dir.glob("**/*.json*"):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Handle both single objects and arrays
                        items = data if isinstance(data, list) else [data]
                        
                        for item in items:
                            if item.get('chunk_id') == doc_id:
                                metadata_dict = {
                                    'title': item.get('title', ''),
                                    'chunk_id': doc_id,
                                    'page_number': item.get('page_number', ''),
                                    'section_level': item.get('section_level', ''),
                                    'file_name': item.get('metadata', {}).get('file_name', ''),
                                    'source_type': 'pdf'
                                }
                                
                                # Add download_url if available
                                if 'metadata' in item and 'download_url' in item['metadata']:
                                    metadata_dict['download_url'] = item['metadata']['download_url']
                                
                                return Document(
                                    page_content=item.get('content', ''),
                                    metadata=metadata_dict
                                )
            except Exception as e:
                logger.error(f"Error searching in {chunks_dir}: {e}")
                continue
        
        logger.warning(f"Document with ID {doc_id} not found")
        return None

# Singleton instance
_rag_model = None

def get_rag_model():
    """Get or create the RAG model singleton"""
    global _rag_model
    if _rag_model is None:
        _rag_model = RAGModel()
    return _rag_model

if __name__ == "__main__":
    # Test the RAG model
    rag = get_rag_model()
    question = "What can we learn from the Gateway mission?"
    
    print("\nTesting with OpenAI:")
    answer_openai = rag.query(question, "openai")
    print(f"\nAnswer (OpenAI): {answer_openai}")
    
    print("\nTesting with Llama:")
    answer_llama = rag.query(question, "llama")
    print(f"\nAnswer (Llama): {answer_llama}") 