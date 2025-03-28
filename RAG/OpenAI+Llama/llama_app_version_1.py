import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# THIS IS AN SLIGHTLY MODIFIED VERSION OF THE APP.PY FILE WHICH FIXED THE ISSUE
# OF THE APP.PY FILE TAKING HALF AN HOUR TO CONDUCT THE VECTOR STORE CREATION.
# THIS VERSION TAKES ONLY A FEW SECONDS TO PERFORM VECTOR CREATION BUT STILL TAKES 
# A LOT OF TIME TO ANSWER THE QUESTION 



# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Start timing
start_time = time.time()
logger.info("Starting application...")

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
logger.info("Environment variables loaded")

# Load CSV data
logger.info("Loading CSV data...")
loader = CSVLoader( 
    file_path="Data Collection/nasa_lessons_learned_jet_propulsion_PROPER.csv",
    source_column="url",
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
    }
)
docs = loader.load()
logger.info(f"Loaded {len(docs)} documents from CSV")
print("1")

# Split text into smaller chunks for better performance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduced from 1000
    chunk_overlap=100  # Reduced from 200
)
print("2")
# Create new documents from the split text
texts = []
for doc in docs:
    # Combine relevant fields into a single text
    combined_text = f"""
    Subject: {doc.metadata.get('subject', '')}
    Abstract: {doc.metadata.get('abstract', '')}
    Driving Event: {doc.metadata.get('driving_event', '')}
    Lessons Learned: {doc.metadata.get('lessons_learned', '')}
    Recommendations: {doc.metadata.get('recommendations', '')}
    """
    # Split the combined text
    splits = text_splitter.split_text(combined_text)
    # Create new documents with the splits
    texts.extend([Document(page_content=split, metadata=doc.metadata) for split in splits])
print("3")
logger.info(f"Created {len(texts)} text chunks")
# Create embeddings
logger.info("Initializing embeddings model...")
embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
)
print("4")
# Create vector store with batching
logger.info("Creating vector store...")
BATCH_SIZE = 32  # Reduced batch size
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

# Process documents in batches with timing
for i in range(0, len(texts), BATCH_SIZE):
    batch_start_time = time.time()
    batch = texts[i:i + BATCH_SIZE]
    current_batch = i//BATCH_SIZE + 1
    
    logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} documents)")
    
    if i == 0:
        db = FAISS.from_documents(batch, embed)
    else:
        db.add_documents(batch)
    
    batch_duration = time.time() - batch_start_time
    docs_per_second = len(batch) / batch_duration
    logger.info(f"Batch {current_batch} completed in {batch_duration:.2f} seconds ({docs_per_second:.2f} docs/second)")

# Save the vector store for future use
db.save_local("faiss_index")

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)
print("5")
logger.info("Vector store created successfully")
# Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are an expert in analyzing NASA lessons learned. Based on the following context, please provide a detailed and well-structured answer to the question.

Context: {context}

Question: {input}

Please provide your answer in the following format:
1. Direct Answer
2. Key Lessons Learned
3. Relevant Recommendations (if any)
""")
print("6")
# Initialize LLM
model = OllamaLLM(model='llama2')
print("7")

combine_docs_chain = create_stuff_documents_chain(model, prompt)

print("8")
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

print("9")

def ask_question(question):
    logger.info(f"Processing question: {question}")
    query_start_time = time.time()
    result = retrieval_chain.invoke({'input': question})
    answer = result["answer"]
    query_duration = time.time() - query_start_time
    
    # Log both the question and answer
    logger.info(f"Question answered in {query_duration:.2f} seconds")
    logger.info("Question: " + question)
    logger.info("Answer: " + answer)
    
    return answer

print("10")

if __name__ == "__main__":
    # Load existing vector store if available
    if os.path.exists("faiss_index"):
        logger.info("Loading existing vector store...")
        db = FAISS.load_local(
            "faiss_index", 
            embed,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
    
    question = "What can we learn from the 2000 HESSI spacecraft overtest incident that severely damaged the spacecraft?"
    print("\nQuestion:", question)
    answer = ask_question(question)
    print("\nAnswer:", answer)
    
    # Log total execution time
    total_duration = time.time() - start_time
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    # ollama pull llama2