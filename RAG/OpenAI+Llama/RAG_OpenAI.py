from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PagedCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import faiss
import os
import pandas as pd
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_openai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Start timing
start_time = time.time()
logger.info("Starting RAG OpenAI application...")

# Load environment variables
logger.info("Loading environment variables...")

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
if os.environ["OPENAI_API_KEY"] is None:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
logger.info("Environment variables loaded successfully")


# Llamaindex global settings for llm and embeddings
logger.info("Initializing LLM and embedding settings...")
EMBED_DIMENSION=512
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
logger.info("LLM and embedding settings initialized")


logger.info("Loading CSV file...")
file_path = 'Data Collection/nasa_lessons_learned_jet_propulsion_PROPER.csv'
data = pd.read_csv(file_path)
logger.info(f"Loaded CSV file with {len(data)} rows")
data.head()

logger.info("Creating FAISS vector store...")
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)
logger.info("FAISS vector store created")

# Set up document reader
logger.info("Setting up document reader...")
csv_reader = PagedCSVReader()
reader = SimpleDirectoryReader(
    input_files=[file_path],
    file_extractor={".csv": csv_reader}
)
docs = reader.load_data()
logger.info(f"Loaded {len(docs)} documents")

# Check a sample chunk
print(docs[0].text)


# Create and run ingestion pipeline
logger.info("Starting document ingestion pipeline...")
pipeline = IngestionPipeline(
    vector_store=vector_store,
    documents=docs
)
nodes = pipeline.run()
logger.info(f"Created {len(nodes)} nodes from documents")

# Create vector store index and query engine
logger.info("Creating vector store index...")
vector_store_index = VectorStoreIndex(nodes)
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
logger.info("Vector store index and query engine created")


def ask_question(question):
    logger.info(f"Processing question: {question}")
    query_start_time = time.time()
    response = query_engine.query(question)
    query_duration = time.time() - query_start_time
    
    logger.info(f"Question answered in {query_duration:.2f} seconds")
    logger.info("Question: " + question)
    logger.info("Answer: " + str(response.response))
    
    return response.response


# Example usage
if __name__ == "__main__":
    question = "What can we learn from the 2000 HESSI spacecraft overtest incident that severely damaged the spacecraft?"
    print("\nQuestion:", question)
    answer = ask_question(question)
    print("\nAnswer:", answer)
    
    # Log total execution time
    total_duration = time.time() - start_time
    logger.info(f"Total execution time: {total_duration:.2f} seconds")