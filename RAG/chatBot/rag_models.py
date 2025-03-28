import os
import time
import logging
from datetime import datetime
import json
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
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

# Define the NASA documents prompt template
nasa_prompt = ChatPromptTemplate.from_template("""
You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

Context: {context}

Question: {input}

Format:
1. Brief Answer
2. Key Points
3. Relevant Mission Details (if applicable)
""")

class RAGModel:
    def __init__(self, chunks_dir="RAG/Section Chunking", index_dir="RAG/chatBot/vector_indices"):
        self.chunks_dir = chunks_dir
        self.index_dir = index_dir
        self.db = None
        self.retriever = None
        
        # Create index directory if it doesn't exist
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        self.embed = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load or create vector store
        self._load_or_create_vector_store()
    
    def _load_chunks(self):
        """Load document chunks from JSON files"""
        logger.info("Loading document chunks...")
        chunks = []
        
        # Get all JSON files in the chunks directory
        json_files = list(Path(self.chunks_dir).glob("*.json*"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.chunks_dir}")
            return chunks
        
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
                            # Create document with content and metadata
                            doc = Document(
                                page_content=item['content'],
                                metadata={
                                    'title': item['title'],
                                    'chunk_id': item.get('chunk_id', ''),
                                    'page_number': item.get('page_number', ''),
                                    'section_level': item.get('section_level', ''),
                                    'file_name': item.get('metadata', {}).get('file_name', '')
                                }
                            )
                            chunks.append(doc)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(chunks)} document chunks")
        return chunks
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one"""
        index_path = Path(self.index_dir) / "nasa_docs_index"
        
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
        
        # Create retriever
        if self.db:
            self.retriever = self.db.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.7
                }
            )
    
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
            else:
                self.db.add_documents(batch)
            
            batch_duration = time.time() - batch_start_time
            docs_per_second = len(batch) / batch_duration
            logger.info(f"Batch {i+1}/{total_batches} completed in {batch_duration:.2f} seconds ({docs_per_second:.2f} docs/second)")
        
        # Save the vector store
        if self.db:
            self.db.save_local(str(Path(self.index_dir) / "nasa_docs_index"))
            logger.info("Vector store created and saved successfully")
    
    def query(self, question, model_name="openai"):
        """Query the RAG model with a question"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "Error: Retriever not initialized. Please try again later."
        
        logger.info(f"Processing question with {model_name}: {question}")
        query_start_time = time.time()
        
        try:
            # Initialize the appropriate LLM based on model_name
            if model_name.lower() == "llama":
                llm = OllamaLLM(
                    model='llama3.1',
                    temperature=0.1,
                    num_ctx=2048,
                    request_timeout=30.0
                )
            else:  # Default to OpenAI
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1
                )
            
            # Create the chain
            combine_docs_chain = create_stuff_documents_chain(llm, nasa_prompt)
            retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
            
            # Execute the query
            result = retrieval_chain.invoke({'input': question})
            answer = result["answer"]
            
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