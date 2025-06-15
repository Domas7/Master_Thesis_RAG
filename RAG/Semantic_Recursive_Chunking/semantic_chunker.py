import os
import json
import logging
from typing import List, Dict, Any, Tuple
import re
from tqdm import tqdm
import pymupdf4llm  # Using pymupdf4llm instead of direct fitz
import fitz  # Still needed for metadata extraction
import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns


# This class is not relevant for the thesis, it is a helper file for the semantic chunking, which is not used in the final implementation. 
# It is kept here for reference in case I will work with it later.

class SemanticChunker:
    
    def __init__(self, 
                 input_dir: str = "../../TESTING_FOLDER_FEW_PDFS", 
                 output_file: str = "semantic_chunks.json",
                 max_chunk_size: int = 1000,
                 similarity_threshold: float = 0.3):
        """
        Initialize the semantic chunker with configuration parameters.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_file: Where to save the resulting chunks as JSON
            max_chunk_size: Maximum characters allowed in each chunk
            similarity_threshold: Minimum similarity score (0-1) to consider sentences related
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("semantic_chunking.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize TF-IDF vectorizer with parameters
        self.vectorizer = TfidfVectorizer(
            min_df=1,                # Include all terms, even if they appear only once
            stop_words='english',    # Remove common English words
            lowercase=True,          # Convert all text to lowercase
            ngram_range=(1, 2)       # Use both single words and pairs of words
        )
        
        # Create a temporary directory for intermediate results
        self.temp_dir = "temp_semantic_chunks"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text content from a PDF file using pymupdf4llm.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string, or empty string if extraction fails
        """
        try:
            # Use pymupdf4llm to convert PDF to markdown
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            
            # Clean up the markdown to get plain text
            # Remove markdown formatting that might interfere with chunking
            text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)  # Remove image references
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Replace links with just their text
            text = re.sub(r'#{1,6}\s+', '', text)  # Remove heading markers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold formatting
            text = re.sub(r'_(.*?)_', r'\1', text)  # Remove italic formatting
            text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code formatting
            
            # Remove page break indicators
            text = re.sub(r'-----+', '\n\n', text)
            
            # Remove page number indicators
            text = re.sub(r'<!-- Page \d+ -->', '', text)
            
            # Normalize whitespace
            text = re.sub(r'\n{3,}', '\n\n\n', text)  # Limit consecutive newlines to 3
            
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Extract title from filename if not in metadata
            if not metadata.get('title'):
                filename = os.path.basename(pdf_path)
                metadata['title'] = os.path.splitext(filename)[0]
            
            # Extract other metadata if available
            result = {
                'title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'file_path': pdf_path,
                'file_name': os.path.basename(pdf_path)
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                'title': 'Unknown',
                'author': 'Unknown',
                'subject': '',
                'keywords': '',
                'file_path': pdf_path,
                'file_name': os.path.basename(pdf_path)
            }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        return sentences
    
    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Compute TF-IDF embeddings for each sentence.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            Matrix of sentence embeddings
        """
        if not sentences:
            return np.array([])
        
        try:
            # Fit and transform the sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            return tfidf_matrix
        except Exception as e:
            self.logger.error(f"Error computing embeddings: {e}")
            return np.array([])
    
    def find_semantic_boundaries(self, 
                                sentences: List[str], 
                                embeddings: np.ndarray) -> List[int]:
        """
        Find natural semantic boundaries between sentences.
        
        Args:
            sentences: List of sentences
            embeddings: TF-IDF embeddings for each sentence
            
        Returns:
            List of indices where semantic boundaries occur
        """
        if len(sentences) <= 1 or embeddings.shape[0] <= 1:
            return []
        
        # Compute similarity between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                embeddings[i:i+1], 
                embeddings[i+1:i+2]
            )[0][0]
            similarities.append(sim)
        
        # Find boundaries where similarity drops below threshold
        boundaries = []
        current_length = 0
        
        for i, sim in enumerate(similarities):
            current_length += len(sentences[i])
            
            # Add a boundary if similarity is low or we've reached max chunk size
            if sim < self.similarity_threshold or current_length >= self.max_chunk_size:
                boundaries.append(i + 1)  # Add the index of the next sentence
                current_length = 0
        
        return boundaries
    
    def create_semantic_chunks(self, 
                              sentences: List[str], 
                              boundaries: List[int]) -> List[str]:
        """
        Create text chunks based on identified semantic boundaries.
        Ensures chunks don't exceed maximum size while preserving semantic meaning.
        """
        if not sentences:
            return []
        
        if not boundaries:
            # If no boundaries found, create chunks of max_chunk_size
            return self.create_size_based_chunks(sentences)
        
        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk = ' '.join(sentences[start_idx:boundary])
            chunks.append(chunk)
            start_idx = boundary
        
        # Add the last chunk
        if start_idx < len(sentences):
            chunk = ' '.join(sentences[start_idx:])
            chunks.append(chunk)
        
        # If any chunk is too large, split it further
        result = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Split large chunks into smaller ones
                sub_sentences = sent_tokenize(chunk)
                sub_chunks = self.create_size_based_chunks(sub_sentences)
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        return result
    
    def create_size_based_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create chunks based on maximum size, respecting sentence boundaries.
        
        Args:
            sentences: List of sentences to chunk
            
        Returns:
            List of text chunks
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed max size and we already have content,
            # finish the current chunk and start a new one
            if current_length + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If a single sentence is longer than max_size, we need to split it
            if len(sentence) > self.max_chunk_size:
                # If we have accumulated content, add it as a chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long sentence into smaller parts at word boundaries
                words = sentence.split()
                current_part = []
                current_part_length = 0
                
                for word in words:
                    # If adding this word would exceed max size and we have content,
                    # finish the current part and start a new one
                    if current_part_length + len(word) + 1 > self.max_chunk_size and current_part:
                        chunks.append(' '.join(current_part))
                        current_part = []
                        current_part_length = 0
                    
                    current_part.append(word)
                    current_part_length += len(word) + 1  # +1 for space
                
                # Add the last part if it has content
                if current_part:
                    chunks.append(' '.join(current_part))
            else:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_pdfs(self) -> None:
        """
        Main processing function that:
        1. Finds all PDFs in input directory
        2. Extracts and processes text from each PDF
        3. Creates semantic chunks
        4. Saves results and generates statistics
        """
        # Get list of PDF files
        pdf_files = []
        
        # Check if input_dir exists
        if not os.path.exists(self.input_dir):
            self.logger.error(f"Input directory '{self.input_dir}' does not exist!")
            return
        
        # Print absolute path for debugging
        abs_path = os.path.abspath(self.input_dir)
        self.logger.info(f"Searching for PDFs in: {abs_path}")
        
        # List files directly in the directory first
        direct_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        if direct_files:
            pdf_files.extend([os.path.join(self.input_dir, f) for f in direct_files])
            self.logger.info(f"Found {len(direct_files)} PDF files directly in the input directory")
        
        # Also walk through subdirectories
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.pdf') and os.path.join(root, file) not in pdf_files:
                    pdf_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(pdf_files)} total PDF files to process")
        
        all_chunks = []
        chunk_counts = {}  # For statistics
        similarity_scores = []  # For statistics
        
        # Statistics tracking
        total_words = 0
        total_chars = 0
        min_words = float('inf')
        max_words = 0
        min_chars = float('inf')
        max_chars = 0
        
        # Process each PDF
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Extract text and metadata
                text = self.extract_text_from_pdf(pdf_path)
                metadata = self.extract_metadata(pdf_path)
                
                if not text:
                    self.logger.warning(f"No text extracted from {pdf_path}")
                    continue
                
                # Split into sentences
                sentences = self.split_into_sentences(text)
                
                if len(sentences) <= 1:
                    self.logger.warning(f"Too few sentences in {pdf_path}")
                    continue
                
                # Compute embeddings
                embeddings = self.compute_sentence_embeddings(sentences)
                
                if embeddings.shape[0] == 0:
                    self.logger.warning(f"Failed to compute embeddings for {pdf_path}")
                    continue
                
                # Find semantic boundaries
                boundaries = self.find_semantic_boundaries(sentences, embeddings)
                
                # Create chunks
                chunks = self.create_semantic_chunks(sentences, boundaries)
                
                # Store statistics
                chunk_counts[os.path.basename(pdf_path)] = len(chunks)
                
                # Compute and store similarity scores for visualization
                if len(sentences) > 1:
                    sims = []
                    for i in range(len(sentences) - 1):
                        sim = cosine_similarity(
                            embeddings[i:i+1], 
                            embeddings[i+1:i+2]
                        )[0][0]
                        sims.append(sim)
                    
                    # Store average similarity for this document
                    if sims:
                        similarity_scores.append((os.path.basename(pdf_path), np.mean(sims)))
                
                # Process each chunk and add to results
                file_chunks = []
                for i, chunk in enumerate(chunks):
                    # Calculate statistics
                    word_count = len(chunk.split())
                    char_count = len(chunk)
                    
                    # Update global statistics
                    total_words += word_count
                    total_chars += char_count
                    min_words = min(min_words, word_count)
                    max_words = max(max_words, word_count)
                    min_chars = min(min_chars, char_count)
                    max_chars = max(max_chars, char_count)
                    
                    chunk_data = {
                        'chunk_id': f"{metadata['file_name']}_{i}",
                        'title': metadata['title'],
                        'author': metadata['author'],
                        'content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'metadata': {
                            **metadata,
                            'statistics': {
                                'word_count': word_count,
                                'character_count': char_count,
                                'average_word_length': char_count / word_count if word_count > 0 else 0
                            }
                        }
                    }
                    all_chunks.append(chunk_data)
                    file_chunks.append(chunk_data)
                
                # Save individual file results
                with open(os.path.join(self.temp_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_chunks.json"), 'w') as f:
                    json.dump(file_chunks, f, indent=2)
                
                self.logger.info(f"Created {len(chunks)} semantic chunks from {pdf_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {e}")
        
        # Save all chunks to JSON
        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        self.logger.info(f"Saved {len(all_chunks)} chunks to {self.output_file}")
        
        # Save statistics and visualizations
        self.save_statistics(chunk_counts, similarity_scores, {
            "total_pdfs": len(pdf_files),
            "total_chunks": len(all_chunks),
            "average_chunks_per_pdf": len(all_chunks)/len(pdf_files) if pdf_files else 0,
            "average_words_per_chunk": total_words / len(all_chunks) if all_chunks else 0,
            "average_chars_per_chunk": total_chars / len(all_chunks) if all_chunks else 0,
            "min_words": min_words if min_words != float('inf') else 0,
            "max_words": max_words,
            "min_chars": min_chars if min_chars != float('inf') else 0,
            "max_chars": max_chars,
            "similarity_threshold": self.similarity_threshold,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def save_statistics(self, 
                        chunk_counts: Dict[str, int],
                        similarity_scores: List[Tuple[str, float]],
                        global_stats: Dict[str, Any]) -> None:
        """Save chunking statistics and visualizations."""
        # Calculate statistics
        total_files = len(chunk_counts)
        total_chunks = sum(chunk_counts.values())
        avg_chunks_per_file = total_chunks / max(1, total_files)
        
        # Create a histogram of chunk counts
        histogram = {}
        for count in chunk_counts.values():
            histogram[count] = histogram.get(count, 0) + 1
        
        # Save statistics to JSON
        statistics = {
            'global': global_stats,
            'per_file': sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True),
            'chunk_count_histogram': histogram,
            'similarity_scores': similarity_scores
        }
        
        with open('semantic_chunking_stats.json', 'w') as f:
            json.dump(statistics, f, indent=2)
        
        self.logger.info(f"Saved statistics to semantic_chunking_stats.json")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Average chunks per file: {avg_chunks_per_file:.2f}")
        
        # Create visualizations
        self.create_visualizations(chunk_counts, similarity_scores)
    
    def create_visualizations(self, 
                             chunk_counts: Dict[str, int],
                             similarity_scores: List[Tuple[str, float]]) -> None:
        """Create visualizations of chunking results."""
        try:
            # Set up the visualization style
            plt.style.use('ggplot')
            
            # 1. Distribution of chunks per document
            plt.figure(figsize=(12, 6))
            counts = list(chunk_counts.values())
            
            if counts:
                sns.histplot(counts, kde=True)
                plt.title('Distribution of Chunks per Document')
                plt.xlabel('Number of Chunks')
                plt.ylabel('Number of Documents')
                plt.savefig('chunks_per_document.png')
                plt.close()
            
            # 2. Top 10 documents with most chunks
            if len(chunk_counts) > 0:
                plt.figure(figsize=(14, 8))
                top_docs = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                if top_docs:
                    names = [doc[0][:20] + '...' if len(doc[0]) > 20 else doc[0] for doc in top_docs]
                    values = [doc[1] for doc in top_docs]
                    
                    plt.barh(names, values)
                    plt.title('Top 10 Documents with Most Chunks')
                    plt.xlabel('Number of Chunks')
                    plt.tight_layout()
                    plt.savefig('top_documents.png')
                    plt.close()
            
            # 3. Average similarity scores
            if similarity_scores:
                plt.figure(figsize=(12, 6))
                scores = [score[1] for score in similarity_scores]
                
                sns.histplot(scores, kde=True)
                plt.axvline(x=self.similarity_threshold, color='red', linestyle='--', 
                           label=f'Threshold ({self.similarity_threshold})')
                plt.title('Distribution of Average Similarity Scores')
                plt.xlabel('Average Similarity Score')
                plt.ylabel('Number of Documents')
                plt.legend()
                plt.savefig('similarity_scores.png')
                plt.close()
            
            self.logger.info("Created visualizations")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    # Create and run the chunker
    chunker = SemanticChunker(
        input_dir="../../TESTING_FOLDER_FEW_PDFS",
        output_file="semantic_chunks.json",
        max_chunk_size=1000,
        similarity_threshold=0.3
    )
    chunker.process_pdfs()