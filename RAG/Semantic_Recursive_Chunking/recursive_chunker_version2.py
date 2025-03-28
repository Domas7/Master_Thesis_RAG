import os
import json
import logging
import re
from tqdm import tqdm
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize  # For sentence tokenization
import pymupdf4llm
import datetime
import time
import signal
from contextlib import contextmanager
import multiprocessing

# Initialize NLTK's sentence tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Timeout exception class
class TimeoutException(Exception):
    pass

# Context manager for timeout
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class RecursiveChunker:
    """
    A class that implements recursive text chunking for PDF documents.
    It splits documents into smaller, overlapping chunks while preserving
    natural text boundaries like paragraphs and sentences.
    """
    
    def __init__(self, 
                 input_dir: str = "NTRS_PDFS_FINAL", 
                 output_file: str = "recursive_chunks.json",
                 max_chunk_size: int = 1000,
                 overlap_size: int = 100,
                 timeout_seconds: int = 60,
                 num_processes: int = None):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_file: Where to save the resulting chunks as JSON
            max_chunk_size: Maximum characters allowed in each chunk
            overlap_size: Number of characters to overlap between chunks for context
            timeout_seconds: Maximum time to spend on a single PDF
            num_processes: Number of parallel processes to use
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.timeout_seconds = timeout_seconds
        self.num_processes = num_processes if num_processes else max(1, multiprocessing.cpu_count() - 1)
        
        # Set up logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("recursive_chunking.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define text separators in order of preference
        # The chunker will try each separator in order until it finds one that works
        self.separators = [
            "\n\n\n",  # Triple newline (major section breaks)
            "\n\n",    # Double newline (paragraph breaks)
            "\n",      # Single newline (line breaks)
            ". ",      # Period-space (sentence boundaries)
            ", ",      # Comma-space (clause boundaries)
            " ",       # Space (word boundaries - last resort)
        ]
        
        # Create a temporary directory for intermediate results
        self.temp_dir = "temp_recursive_chunks"
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
            
            # Clean up the markdown text
            # Remove page markers
            markdown_text = re.sub(r'<!-- Page \d+ -->', '', markdown_text)
            # Remove horizontal rules (often used as page breaks)
            markdown_text = re.sub(r'-----+', '', markdown_text)
            # Remove image references
            markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
            # Remove table markers
            markdown_text = re.sub(r'\|.*?\|', '', markdown_text)
            
            # Remove multiple consecutive newlines (more than 3)
            markdown_text = re.sub(r'\n{4,}', '\n\n\n', markdown_text)
            
            return markdown_text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata (title, author, etc.) from a PDF file.
        Falls back to filename if metadata is not available.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata fields
        """
        try:
            # Use pymupdf4llm to get metadata
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Use filename as title if no title in metadata
            if not metadata.get('title'):
                filename = os.path.basename(pdf_path)
                metadata['title'] = os.path.splitext(filename)[0]
            
            # Construct metadata dictionary with fallbacks for missing fields
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
    
    def recursive_split(self, text: str, separators: List[str], max_size: int) -> List[str]:
        """
        Recursively split text into chunks using a hierarchy of separators.
        Tries to split at natural boundaries (sections, paragraphs, sentences)
        before falling back to character-based splits.
        
        Args:
            text: Text to split
            separators: List of separators to try (in order of preference)
            max_size: Maximum size for each chunk
            
        Returns:
            List of text chunks
        """
        # Base case: if text is already small enough, return it as a single chunk
        if len(text) <= max_size:
            return [text]
        
        # If we've exhausted all separators, force split at max_size
        if not separators:
            # Try to find a space near max_size to avoid splitting words
            split_point = max_size
            while split_point > max_size - 50 and split_point > 0:
                if text[split_point] == ' ':
                    break
                split_point -= 1
            
            # If we couldn't find a space, just split at max_size
            if split_point == 0:
                split_point = max_size
                
            return [text[:split_point], text[split_point:]]
        
        # Try splitting with the current separator
        separator = separators[0]
        parts = text.split(separator)
        
        # If splitting didn't work (text remained as one piece),
        # try the next separator in the list
        if len(parts) == 1:
            return self.recursive_split(text, separators[1:], max_size)
        
        # Recombine parts into chunks that don't exceed max_size
        chunks = []
        current_chunk = parts[0]
        
        for part in parts[1:]:
            # Check if adding the next part would exceed max_size
            if len(current_chunk) + len(separator) + len(part) > max_size:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += separator + part
        
        # Add the final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Recursively split any chunks that are still too large
        result = []
        for chunk in chunks:
            if len(chunk) > max_size:
                result.extend(self.recursive_split(chunk, separators[1:], max_size))
            else:
                result.append(chunk)
        
        return result
    
    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks to maintain context across chunk boundaries.
        Takes the end of the previous chunk and prepends it to the current chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with overlap added
        """
        if not chunks or len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Take overlap_size characters from end of previous chunk
            overlap = prev_chunk[-min(self.overlap_size, len(prev_chunk)):]
            
            # Add overlap to beginning of current chunk
            result.append(overlap + current_chunk)
        
        return result
    
    def process_single_pdf(self, pdf_file: str):
        """
        Process a single PDF file and return its chunks and statistics.
        
        Args:
            pdf_file: Name of the PDF file to process
            
        Returns:
            Tuple of (chunks, error_info, statistics)
        """
        pdf_path = os.path.join(self.input_dir, pdf_file)
        file_chunks = []
        stats = {
            "words": 0,
            "chars": 0,
            "min_words": float('inf'),
            "max_words": 0,
            "min_chars": float('inf'),
            "max_chars": 0,
            "chunks": 0,
            "status": "success"
        }
        error_info = None
        
        try:
            # Use timeout context manager
            with time_limit(self.timeout_seconds):
                start_time = time.time()
                
                # Extract content and metadata
                text = self.extract_text_from_pdf(pdf_path)
                metadata = self.extract_metadata(pdf_path)
                
                if not text:
                    error_info = f"No text extracted from {pdf_path}"
                    stats["status"] = "error"
                    return [], [error_info], stats
                
                # Create chunks and add overlap
                chunks = self.recursive_split(text, self.separators, self.max_chunk_size)
                chunks = self.create_overlapping_chunks(chunks)
                
                # Add chunks to result with metadata
                for i, chunk in enumerate(chunks):
                    # Calculate statistics
                    word_count = len(chunk.split())
                    char_count = len(chunk)
                    
                    # Update statistics
                    stats["words"] += word_count
                    stats["chars"] += char_count
                    stats["min_words"] = min(stats["min_words"], word_count)
                    stats["max_words"] = max(stats["max_words"], word_count)
                    stats["min_chars"] = min(stats["min_chars"], char_count)
                    stats["max_chars"] = max(stats["max_chars"], char_count)
                    stats["chunks"] += 1
                    
                    chunk_data = {
                        'chunk_id': f"{metadata['file_name']}_{i}",
                        'title': metadata['title'],
                        'author': metadata['author'],
                        'content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'metadata': {
                            'file_name': metadata['file_name'],
                            'file_path': metadata['file_path'],
                            'statistics': {
                                'word_count': word_count,
                                'character_count': char_count,
                                'average_word_length': char_count / word_count if word_count > 0 else 0,
                                'processing_time_seconds': time.time() - start_time
                            }
                        }
                    }
                    file_chunks.append(chunk_data)
                
                # Save individual file results
                with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_chunks.json"), 'w') as f:
                    json.dump(file_chunks, f, indent=2)
                
                print(f"Processed {pdf_file} in {time.time() - start_time:.2f} seconds - created {len(chunks)} chunks")
        
        except TimeoutException:
            error_info = f"Timeout: exceeded {self.timeout_seconds} seconds"
            stats["status"] = "timeout"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_timeout.txt"), 'w') as f:
                f.write(f"Skipped due to timeout (exceeded {self.timeout_seconds} seconds)")
        
        except Exception as e:
            error_info = f"Error: {str(e)}"
            stats["status"] = "error"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_error.txt"), 'w') as f:
                f.write(f"Error: {str(e)}")
        
        return file_chunks, [error_info] if error_info else [], stats
    
    def process_pdfs(self) -> None:
        """
        Main processing function that:
        1. Finds all PDFs in the input directory
        2. Extracts text and metadata from each PDF
        3. Creates chunks using recursive splitting
        4. Adds overlap between chunks
        5. Saves results and statistics
        
        Uses multiprocessing for parallel execution.
        """
        # Find all PDF files in the input directory
        pdf_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.basename(os.path.join(root, file)))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.logger.info(f"Using {self.num_processes} processes for parallel processing")
        
        # Statistics tracking
        all_chunks = []
        skipped_files = []
        total_stats = {
            "total_words": 0,
            "total_chars": 0,
            "min_words": float('inf'),
            "max_words": 0,
            "min_chars": float('inf'),
            "max_chars": 0,
            "total_chunks": 0
        }
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Process files in parallel and show progress with tqdm
            results = list(tqdm(
                pool.imap(self.process_single_pdf, pdf_files),
                total=len(pdf_files),
                desc="Processing PDFs"
            ))
        
        # Collect results
        for i, (chunks, errors, stats) in enumerate(results):
            # Add chunks to the overall list
            all_chunks.extend(chunks)
            
            # Update statistics
            if stats["status"] == "success":
                total_stats["total_words"] += stats["words"]
                total_stats["total_chars"] += stats["chars"]
                total_stats["min_words"] = min(total_stats["min_words"], stats["min_words"]) if stats["min_words"] != float('inf') else total_stats["min_words"]
                total_stats["max_words"] = max(total_stats["max_words"], stats["max_words"])
                total_stats["min_chars"] = min(total_stats["min_chars"], stats["min_chars"]) if stats["min_chars"] != float('inf') else total_stats["min_chars"]
                total_stats["max_chars"] = max(total_stats["max_chars"], stats["max_chars"])
                total_stats["total_chunks"] += stats["chunks"]
            else:
                skipped_files.append(pdf_files[i])
            
            # Save intermediate results periodically
            if (i + 1) % 10 == 0 or i == len(results) - 1:
                self.logger.info(f"Saving intermediate results after processing {i + 1}/{len(pdf_files)} files")
                with open(f"{self.output_file}.partial", 'w') as f:
                    json.dump(all_chunks, f, indent=2)
                
                # Also save current statistics
                current_stats = {
                    "total_pdfs_processed": i + 1,
                    "total_chunks": total_stats["total_chunks"],
                    "skipped_files": len(skipped_files),
                    "skipped_file_list": skipped_files,
                    "average_chunks_per_pdf": total_stats["total_chunks"]/(i + 1 - len(skipped_files)) if (i + 1 - len(skipped_files)) > 0 else 0,
                    "average_words_per_chunk": total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                    "average_chars_per_chunk": total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                    "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
                    "max_words": total_stats["max_words"],
                    "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
                    "max_chars": total_stats["max_chars"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "progress_percentage": ((i + 1) / len(pdf_files)) * 100
                }
                with open("recursive_chunk_statistics.partial.json", 'w') as f:
                    json.dump(current_stats, f, indent=2)
        
        # Save final results
        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        # Calculate averages
        avg_words = total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        avg_chars = total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        
        # Log overall statistics
        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"Total PDFs processed: {len(pdf_files) - len(skipped_files)}")
        self.logger.info(f"Total PDFs skipped: {len(skipped_files)}")
        self.logger.info(f"Total chunks created: {total_stats['total_chunks']}")
        if pdf_files and len(pdf_files) > len(skipped_files):
            self.logger.info(f"Average chunks per PDF: {total_stats['total_chunks']/(len(pdf_files) - len(skipped_files)):.1f}")
        
        self.logger.info(f"\nChunk Statistics:")
        self.logger.info(f"Average words per chunk: {avg_words:.1f}")
        self.logger.info(f"Average characters per chunk: {avg_chars:.1f}")
        self.logger.info(f"Word count range: {total_stats['min_words']} to {total_stats['max_words']}")
        self.logger.info(f"Character count range: {total_stats['min_chars']} to {total_stats['max_chars']}")
        
        # Save overall statistics to a separate JSON file
        statistics = {
            "total_pdfs": len(pdf_files),
            "total_pdfs_processed": len(pdf_files) - len(skipped_files),
            "total_pdfs_skipped": len(skipped_files),
            "skipped_file_list": skipped_files,
            "total_chunks": total_stats["total_chunks"],
            "average_chunks_per_pdf": total_stats["total_chunks"]/(len(pdf_files) - len(skipped_files)) if (len(pdf_files) - len(skipped_files)) > 0 else 0,
            "average_words_per_chunk": avg_words,
            "average_chars_per_chunk": avg_chars,
            "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
            "max_words": total_stats["max_words"],
            "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
            "max_chars": total_stats["max_chars"],
            "timestamp": datetime.datetime.now().isoformat(),
            "chunking_method": "recursive",
            "max_chunk_size": self.max_chunk_size,
            "overlap_size": self.overlap_size
        }
        
        with open("recursive_chunk_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)

if __name__ == "__main__":
    chunker = RecursiveChunker()
    chunker.process_pdfs()