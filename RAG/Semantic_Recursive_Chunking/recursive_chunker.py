import os
import json
import logging
from typing import List, Dict, Any
import re
from tqdm import tqdm
import pymupdf4llm  # Using pymupdf4llm instead of direct fitz
import fitz  # Still needed for metadata extraction
import datetime

class RecursiveChunker:
    """
    A class that implements recursive text chunking for PDF documents.
    It splits documents into smaller, overlapping chunks while preserving
    natural text boundaries like paragraphs and sentences.
    """
    
    def __init__(self, 
                 input_dir: str = "../../TESTING_FOLDER_FEW_PDFS",  # Updated path to go up two directories
                 output_file: str = "recursive_chunks.json",
                 max_chunk_size: int = 1000,
                 overlap_size: int = 100):  
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_file: Where to save the resulting chunks as JSON
            max_chunk_size: Maximum characters allowed in each chunk
            overlap_size: Number of characters to overlap between chunks for context
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
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
        """
        Extract metadata (title, author, etc.) from a PDF file.
        Falls back to filename if metadata is not available.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata fields
        """
        try:
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
        Recursively split text into chunks using natural separators.
        Always respects word boundaries and tries to respect sentence boundaries.
        
        Args:
            text: Text to split
            separators: List of separators to try, in order of preference
            max_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # If text is already small enough, return it as a single chunk
        if len(text) <= max_size:
            return [text]
        
        # Try each separator in order
        for separator in separators:
            # Skip empty separators
            if not separator:
                continue
            
            # If text doesn't contain this separator, try the next one
            if separator not in text:
                continue
            
            # Find the best position to split using this separator
            # We want to split as close to max_size as possible, but not over
            best_pos = -1
            
            # Find all occurrences of the separator
            positions = [match.start() for match in re.finditer(re.escape(separator), text)]
            
            # Find the position closest to max_size without going over
            for pos in positions:
                if pos <= max_size and pos > best_pos:
                    best_pos = pos
            
            # If we found a good position, split there
            if best_pos > 0:
                # Split at the end of the separator
                split_pos = best_pos + len(separator)
                
                # Make sure we're not splitting in the middle of a word
                # If we are at a space separator, this is already handled
                if separator.strip() and split_pos < len(text):
                    # If we're not at a space and the next character is not a space,
                    # we might be in the middle of a word - find the next space
                    if text[split_pos-1] != ' ' and text[split_pos] != ' ':
                        # Find the next space
                        next_space = text.find(' ', split_pos)
                        if next_space != -1 and next_space - split_pos < 20:  # Don't go too far
                            split_pos = next_space + 1
                
                # Create chunks
                first_chunk = text[:split_pos].strip()
                rest = text[split_pos:].strip()
                
                # Recursively split the rest
                if rest:
                    return [first_chunk] + self.recursive_split(rest, separators, max_size)
                else:
                    return [first_chunk]
        
        # If we get here, we couldn't find a good separator
        # Fall back to splitting at the max_size, but ensure we don't split words
        if len(text) > max_size:
            # Find the last space before max_size
            last_space = text[:max_size].rfind(' ')
            if last_space > max_size // 2:  # Only use if it's not too far back
                split_pos = last_space + 1
            else:
                # If no good space found, find the next space after max_size
                next_space = text.find(' ', max_size)
                if next_space != -1 and next_space - max_size < 20:  # Don't go too far
                    split_pos = next_space + 1
                else:
                    # Last resort: split at max_size
                    split_pos = max_size
            
            first_chunk = text[:split_pos].strip()
            rest = text[split_pos:].strip()
            
            if rest:
                return [first_chunk] + self.recursive_split(rest, separators, max_size)
            else:
                return [first_chunk]
        
        # If all else fails, return the text as a single chunk
        return [text]
    
    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks to provide context between chunks.
        Ensures that chunks don't start in the middle of words.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of overlapping text chunks
        """
        if not chunks or len(chunks) <= 1:
            return chunks
        
        result = []
        for i in range(len(chunks)):
            if i == 0:
                # First chunk remains unchanged
                result.append(chunks[i])
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                
                # Calculate overlap size, but don't exceed the previous chunk's length
                overlap_size = min(self.overlap_size, len(prev_chunk))
                
                # Get the overlap text from the end of the previous chunk
                overlap_text = prev_chunk[-overlap_size:]
                
                # Make sure we don't start in the middle of a word
                if overlap_text and not overlap_text[0].isspace() and i > 0:
                    # Find the first space in the overlap
                    first_space = overlap_text.find(' ')
                    if first_space != -1:
                        # Adjust the overlap to start at the first space
                        overlap_text = overlap_text[first_space+1:]
                    else:
                        # If no space found, don't use overlap
                        overlap_text = ""
                
                # Create the new chunk with overlap
                new_chunk = overlap_text + current_chunk
                result.append(new_chunk)
        
        return result
    
    def process_pdfs(self) -> None:
        """
        Main processing function that:
        1. Finds all PDFs in the input directory
        2. Extracts text and metadata from each PDF
        3. Creates chunks using recursive splitting
        4. Adds overlap between chunks
        5. Saves results and statistics
        """
        # Find all PDF files in the input directory
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
        
        all_chunks = []  # Store all chunks from all PDFs
        chunk_counts = {}  # Track number of chunks per file
        
        # Statistics tracking
        total_words = 0
        total_chars = 0
        min_words = float('inf')
        max_words = 0
        min_chars = float('inf')
        max_chars = 0
        
        # Process each PDF with progress bar
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Extract content and metadata
                text = self.extract_text_from_pdf(pdf_path)
                metadata = self.extract_metadata(pdf_path)
                
                if not text:
                    self.logger.warning(f"No text extracted from {pdf_path}")
                    continue
                
                # Create chunks and add overlap
                chunks = self.recursive_split(text, self.separators, self.max_chunk_size)
                chunks = self.create_overlapping_chunks(chunks)
                
                # Store statistics
                chunk_counts[os.path.basename(pdf_path)] = len(chunks)
                
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
                
                self.logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {e}")
        
        # Save all chunks to JSON file
        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        self.logger.info(f"Saved {len(all_chunks)} chunks to {self.output_file}")
        
        # Generate and save statistics
        self.save_statistics(chunk_counts, {
            "total_pdfs": len(pdf_files),
            "total_chunks": len(all_chunks),
            "average_chunks_per_pdf": len(all_chunks)/len(pdf_files) if pdf_files else 0,
            "average_words_per_chunk": total_words / len(all_chunks) if all_chunks else 0,
            "average_chars_per_chunk": total_chars / len(all_chunks) if all_chunks else 0,
            "min_words": min_words if min_words != float('inf') else 0,
            "max_words": max_words,
            "min_chars": min_chars if min_chars != float('inf') else 0,
            "max_chars": max_chars,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def save_statistics(self, chunk_counts: Dict[str, int], global_stats: Dict[str, Any]) -> None:
        """
        Save processing statistics to a JSON file.
        
        Args:
            chunk_counts: Dictionary mapping filenames to number of chunks
            global_stats: Dictionary with global statistics
        """
        # Calculate per-file statistics
        per_file_stats = []
        for filename, count in chunk_counts.items():
            per_file_stats.append({
                "file_name": filename,
                "chunks": count
            })
        
        # Sort files by number of chunks (descending)
        per_file_stats.sort(key=lambda x: x["chunks"], reverse=True)
        
        # Combine all statistics
        statistics = {
            "global": global_stats,
            "per_file": per_file_stats
        }
        
        # Save to JSON file
        with open("recursive_chunk_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)
        
        self.logger.info("Saved chunking statistics to recursive_chunk_statistics.json")

# Entry point of the script
if __name__ == "__main__":
    # Create and run the chunker with default parameters
    chunker = RecursiveChunker(
        input_dir="../../TESTING_FOLDER_FEW_PDFS",
        output_file="recursive_chunks.json",
        max_chunk_size=1000,  # Maximum characters per chunk
        overlap_size=100      # Characters to overlap between chunks
    )
    chunker.process_pdfs()