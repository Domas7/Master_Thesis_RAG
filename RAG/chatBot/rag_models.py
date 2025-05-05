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
import requests
import openai
import threading

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

# Load environment variables from .env file if it exists
load_dotenv()

# Try to get API keys from environment variables first
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

# If not found in environment, try to get from Streamlit secrets
try:
    if not openai_api_key and 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
    if not langchain_api_key and 'LANGCHAIN_API_KEY' in st.secrets:
        langchain_api_key = st.secrets['LANGCHAIN_API_KEY']
except Exception as e:
    logger.warning(f"Could not access Streamlit secrets: {e}")

# Set the API keys if found
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key

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
        self.model_loading = False
        self.model_loaded = False
        
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
                # Get Ollama service URL from environment or use default
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                logger.info(f"Connecting to Ollama at: {ollama_base_url}")
                
                # Test connection to Ollama
                try:
                    response = requests.get(f"{ollama_base_url}/api/tags")
                    if response.status_code != 200:
                        raise Exception(f"Ollama API returned status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    raise Exception(f"Could not connect to Ollama service at {ollama_base_url}: {str(e)}")
                
                self.llm = OllamaLLM(
                    base_url=ollama_base_url,
                    model="mistral",
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
                logger.info("Mistral model loaded successfully in background")
            except Exception as e:
                logger.error(f"Error loading Mistral model in background: {e}")
                self.llm = None
                # Don't fall back to OpenAI, just set llm to None
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
                    
                    content = "\n\n".join(content_parts)
                    
                    # Create metadata
                    metadata_dict = {
                        'title': row.get('subject', 'NASA Lesson Learned'),
                        'url': row.get('url', ''),
                        'source_type': 'lessons_learned',
                        'mission_directorate': row.get('mission_directorate', '')
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
                    "k": 5,
                    "score_threshold": 0.7
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
                if self.llm is None:
                    if self.model_loading:
                        return "The model is still loading. Please try again in a few moments."
                    else:
                        # Start loading the model if it hasn't been loaded yet
                        self._start_background_model_loading()
                        return "The model is being loaded. Please try again in a few moments."
                
                # Create a more concise prompt for faster processing
                llama_prompt = ChatPromptTemplate.from_template("""
                Based on the context, provide a concise answer to the question.

                Context: {context}
                Question: {input}

                Format your response as:
                1. Brief Answer
                2. Key Points
                """)
                
                # Use the concise prompt for faster processing
                combine_docs_chain = create_stuff_documents_chain(self.llm, llama_prompt)
                
            else:  # Default to OpenAI
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Use the full NASA prompt for OpenAI
                combine_docs_chain = create_stuff_documents_chain(llm, nasa_prompt)
            
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
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

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