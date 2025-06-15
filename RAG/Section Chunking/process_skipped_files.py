import os
import json
import logging
import time
import datetime
import multiprocessing
from typing import List, Dict, Any, Tuple
from functools import partial
import re
from tqdm import tqdm
import pymupdf4llm
import signal
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import traceback


# This file is not relevant for the thesis. When working with chunking, some files were skipped, if they took too long and similarly,
# The skipped files were logged and this file was created to process the skipped files, which follows THE SAME structure as the main processing file
# just with more lenient approach, meaning give it more time, more tries and so on.. 

# Timeout exception and handler
class TimeoutException(Exception):
    pass

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

@dataclass
class TextProperties:
    """Store text properties from PDF"""
    text: str
    font_size: float
    is_bold: bool
    bbox: tuple  # Bounding box coordinates (x0, y0, x1, y1, is_italic)
    page_num: int  # Page number

@dataclass
class Section:
    """Represent a document section"""
    title: str
    content: str
    level: int  # Hierarchy level (1 for main sections, 2 for subsections, etc.)
    page_number: int

class SkippedFilesChunker:
    """Process only the skipped files from the previous run"""
    
    def __init__(self, 
                 input_dir: str = "RAG/NTRS_PDFS_CONFERENCE_GLOBAL",
                 output_file: str = "RAG/Section Chunking/skipped_files_section_chunks_CHECKTHIS3333_1591.json",
                 stats_file: str = "RAG/Section Chunking/skipped_chunk_statistics_33333_1591.json",
                 timeout_seconds: int = 500,
                 num_processes: int = None,
                 log_file: str = "RAG/NASA_Technical_Server/nasa_pdf_processing_DONE.log"): 
        self.input_dir = input_dir
        self.output_file = output_file
        self.stats_file = stats_file
        self.timeout_seconds = timeout_seconds
        self.num_processes = num_processes if num_processes else max(1, multiprocessing.cpu_count() - 1)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("skipped_files_chunking.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create a temporary directory for intermediate results if it doesn't exist
        self.temp_dir = "temp_skipped_chunks"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load download URLs from log file
        self.log_file = log_file
        self.download_urls = self.extract_download_urls_from_log(log_file)
        self.logger.info(f"Loaded {len(self.download_urls)} download URLs from log file")
    
    def process_pdfs(self):
        # Load the list of skipped files from the chunk_statistics.json
        try:
            with open("RAG/Section Chunking/skipped_chunk_statistics_1591.json", 'r') as f:
                stats = json.load(f)
                skipped_files = stats.get("still_skipped_file_list", [])
        except Exception as e:
            self.logger.error(f"Error loading skipped files list: {e}")
            return
        
        self.logger.info(f"Found {len(skipped_files)} skipped files to process")
        
        # Filter to only include files that exist in the input directory.
        pdf_files = []
        for file in skipped_files:
            file_path = os.path.join(self.input_dir, file)
            if os.path.exists(file_path):
                pdf_files.append(file)
            else:
                self.logger.warning(f"Skipped file not found: {file}")
        
        self.logger.info(f"Found {len(pdf_files)} skipped files in the input directory")
        
        # Process PDFs in parallel
        process_func = partial(self.process_single_pdf)
        
        all_chunks = []
        still_skipped_files = []
        
        # Initialize statistics
        total_stats = {
            "total_words": 0,
            "total_chars": 0,
            "total_chunks": 0,
            "min_words": float('inf'),
            "max_words": 0,
            "min_chars": float('inf'),
            "max_chars": 0
        }
        
        # Process files in batches to save intermediate results
        batch_size = 100
        
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size}")
            
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                batch_results = list(tqdm(pool.imap(process_func, batch), total=len(batch), desc="Processing PDFs"))
            
            # Process results
            for file_chunks, errors, stats in batch_results:
                if errors:
                    still_skipped_files.append(stats["file_name"])
                    self.logger.warning(f"Skipped {stats['file_name']}: {errors[0]}")
                else:
                    # Update statistics
                    total_stats["total_words"] += stats["word_count"]
                    total_stats["total_chars"] += stats["char_count"]
                    total_stats["total_chunks"] += len(file_chunks)
                    
                    if stats["word_count"] > 0:
                        total_stats["min_words"] = min(total_stats["min_words"], min(chunk["metadata"]["statistics"]["word_count"] for chunk in file_chunks))
                        total_stats["max_words"] = max(total_stats["max_words"], max(chunk["metadata"]["statistics"]["word_count"] for chunk in file_chunks))
                    
                    if stats["char_count"] > 0:
                        total_stats["min_chars"] = min(total_stats["min_chars"], min(chunk["metadata"]["statistics"]["character_count"] for chunk in file_chunks))
                        total_stats["max_chars"] = max(total_stats["max_chars"], max(chunk["metadata"]["statistics"]["character_count"] for chunk in file_chunks))
                    
                    all_chunks.extend(file_chunks)
            
            # Save intermediate results
            with open(f"{self.output_file}.partial", 'w') as f:
                json.dump(all_chunks, f, indent=2)
            
            # Save intermediate statistics
            current_stats = {
                "processed_so_far": i + len(batch),
                "total_pdfs": len(pdf_files),
                "total_chunks": total_stats["total_chunks"],
                "skipped_files": len(still_skipped_files),
                "still_skipped_file_list": still_skipped_files,
                "average_chunks_per_pdf": total_stats["total_chunks"]/(i + len(batch) - len(still_skipped_files)) if (i + len(batch) - len(still_skipped_files)) > 0 else 0,
                "average_words_per_chunk": total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                "average_chars_per_chunk": total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
                "max_words": total_stats["max_words"],
                "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
                "max_chars": total_stats["max_chars"]
            }
            
            with open(f"{self.stats_file}.partial", 'w') as f:
                json.dump(current_stats, f, indent=2)
        
        # Save final results
        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        # Calculate averages
        avg_words = total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        avg_chars = total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        
        # Log overall statistics
        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"Total PDFs processed: {len(pdf_files) - len(still_skipped_files)}")
        self.logger.info(f"Total PDFs still skipped: {len(still_skipped_files)}")
        self.logger.info(f"Total chunks created: {total_stats['total_chunks']}")
        
        if pdf_files and len(pdf_files) > len(still_skipped_files):
            self.logger.info(f"Average chunks per PDF: {total_stats['total_chunks']/(len(pdf_files) - len(still_skipped_files)):.1f}")
        
        self.logger.info(f"\nChunk Statistics:")
        self.logger.info(f"Average words per chunk: {avg_words:.1f}")
        self.logger.info(f"Average characters per chunk: {avg_chars:.1f}")
        self.logger.info(f"Word count range: {total_stats['min_words']} to {total_stats['max_words']}")
        self.logger.info(f"Character count range: {total_stats['min_chars']} to {total_stats['max_chars']}")
        
        # Save overall statistics to a separate JSON file
        statistics = {
            "total_pdfs_attempted": len(pdf_files),
            "total_pdfs_processed": len(pdf_files) - len(still_skipped_files),
            "total_pdfs_still_skipped": len(still_skipped_files),
            "still_skipped_file_list": still_skipped_files,
            "total_chunks": total_stats["total_chunks"],
            "average_chunks_per_pdf": total_stats["total_chunks"]/(len(pdf_files) - len(still_skipped_files)) if (len(pdf_files) - len(still_skipped_files)) > 0 else 0,
            "average_words_per_chunk": avg_words,
            "average_chars_per_chunk": avg_chars,
            "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
            "max_words": total_stats["max_words"],
            "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
            "max_chars": total_stats["max_chars"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
    
    def process_single_pdf(self, pdf_file: str) -> Tuple[List[Dict], List[str], Dict]:
        """Process a single PDF file and extract sections"""
        pdf_path = os.path.join(self.input_dir, pdf_file)
        stats = {
            "file_name": pdf_file,
            "status": "success",
            "sections_extracted": 0,
            "word_count": 0,
            "char_count": 0,
            "processing_time": 0
        }
        
        file_chunks = []
        error_info = None
        
        try:
            start_time = time.time()
            
            with time_limit(self.timeout_seconds):
                # Try up to 3 times to extract with pymupdf4llm
                markdown_text = None
                for attempt in range(3):
                    try:
                        markdown_text = pymupdf4llm.to_markdown(pdf_path)
                        break  # Success, exit the retry loop
                    except TimeoutException:
                        if attempt < 2:  # If not the last attempt
                            self.logger.warning(f"Attempt {attempt+1} timed out for {pdf_file}, retrying...")
                            time.sleep(1)  # Brief pause before retry
                        else:
                            raise  # Re-raise on last attempt
                    except Exception as e:
                        if attempt < 2:  # If not the last attempt
                            self.logger.warning(f"Attempt {attempt+1} failed for {pdf_file}: {str(e)}, retrying...")
                            time.sleep(1)  # Brief pause before retry
                        else:
                            raise  # Re-raise on last attempt
                
                if not markdown_text:
                    raise Exception("All extraction attempts failed")
                
                # Extract sections from the markdown
                sections = self.extract_sections_from_markdown(markdown_text, pdf_path)
                
                # Post-process sections
                sections = self.post_process_sections(sections)
                
                # Verify sections
                self.verify_sections(sections, pdf_path)
                
                stats["sections_extracted"] = len(sections)
                
                # Create chunks from sections
                for i, section in enumerate(sections):
                    # Calculate statistics
                    word_count = len(section.content.split())
                    char_count = len(section.content)
                    
                    stats["word_count"] += word_count
                    stats["char_count"] += char_count
                    
                    # Create the chunk
                    chunk = {
                        'chunk_id': f"{pdf_file}_{i}",
                        'title': section.title,
                        'content': section.content,
                        'page_number': section.page_number,
                        'section_level': section.level,
                        'section_number': i + 1,
                        'total_sections': len(sections),
                        'metadata': {
                            'file_name': pdf_file,
                            'file_path': pdf_path,
                            'download_url': self.get_download_url(pdf_file),
                            'statistics': {
                                'word_count': word_count,
                                'character_count': char_count,
                                'average_word_length': char_count / word_count if word_count > 0 else 0,
                                'processing_time_seconds': time.time() - start_time
                            }
                        }
                    }
                    
                    file_chunks.append(chunk)
                
                stats["processing_time"] = time.time() - start_time
                self.logger.info(f"Processed {pdf_file} in {stats['processing_time']:.2f} seconds")
        
        except TimeoutException:
            error_info = f"Timeout: exceeded {self.timeout_seconds} seconds"
            stats["status"] = "timeout"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_timeout.txt"), 'w') as f:
                f.write(f"Skipped due to timeout (exceeded {self.timeout_seconds} seconds)")
        
        except Exception as e:
            error_info = f"Error: {str(e)}"
            stats["status"] = "error"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_error.txt"), 'w') as f:
                f.write(f"Error: {str(e)}\nTraceback: {traceback.format_exc()}")
        
        return file_chunks, [error_info] if error_info else [], stats
    
    def extract_sections_from_markdown(self, markdown_text: str, pdf_path: str) -> List[Section]:
        sections = []
        current_section = None
        current_content = []
        current_level = 0
        page_number = 1
        
        # Add a default section in case no sections are found
        default_section = Section(
            title="Document Content",
            level=0,
            content="",
            page_number=1
        )
        
        try:
            lines = markdown_text.split('\n')
            
            for line in lines:
                # Check for page breaks
                if line.startswith('---') and 'Page' in line:
                    try:
                        page_number = int(line.split('Page')[1].strip().split()[0])
                    except:
                        # If page number extraction fails, just increment
                        page_number += 1
                    continue
                    
                # Check for section headers (# Header, ## Subheader, etc.)
                header_match = re.match(r'^(#+)\s+(.+)$', line)
                
                if header_match:
                    # If we have a current section, save it before starting a new one
                    if current_section:
                        current_section.content = '\n'.join(current_content).strip()
                        if current_section.content:  # Only add non-empty sections
                            sections.append(current_section)
                    
                    # Start a new section
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    
                    current_section = Section(
                        title=title,
                        level=level,
                        content="",
                        page_number=page_number
                    )
                    current_content = []
                    current_level = level
                else:
                    # Add content to current section
                    if current_section:
                        current_content.append(line)
                    else:
                        # If no section has been created yet, add to default section
                        default_section.content += line + "\n"
            
            # Don't forget to add the last section
            if current_section:
                current_section.content = '\n'.join(current_content).strip()
                if current_section.content:  # Only add non-empty sections
                    sections.append(current_section)
            
            # If no sections were found, use the default section
            if not sections and default_section.content.strip():
                sections.append(default_section)
                
            return sections
        
        except Exception as e:
            self.logger.error(f"Error extracting sections from {pdf_path}: {str(e)}")
            # Return a single section with all content if extraction fails
            if default_section.content.strip():
                return [default_section]
            else:
                # Create a minimal section with error information
                error_section = Section(
                    title="Error Processing Document",
                    level=0,
                    content=f"Failed to extract sections: {str(e)}\n\nPartial content: {markdown_text[:500]}...",
                    page_number=1
                )
                return [error_section]
    
    def post_process_sections(self, sections: List[Section]) -> List[Section]:
        """Post-process sections to clean up and merge if needed"""
        if not sections:
            return []
        
        # Handle empty sections
        sections = [s for s in sections if s.content.strip()]
        
        if not sections:
            # If all sections were empty, create a default section
            return [Section(
                title="Document Content",
                level=0,
                content="No extractable content found in document",
                page_number=1
            )]
        
        # Check for very short sections
        for section in sections:
            word_count = len(section.content.split())
            if word_count < 10 and section.title != "Abstract" and section.title != "References":
                self.logger.warning(f"Very short section '{section.title}' with only {word_count} words")
        
        return sections
    
    def verify_sections(self, sections: List[Section], pdf_path: str) -> None:
        # If no sections were found, create a default section
        if not sections:
            self.logger.warning(f"No sections found in {pdf_path}, creating default section")
            sections.append(Section(
                title="Document Content",
                level=0,
                content="No sections detected in document",
                page_number=1
            ))
            return
        
        # Check for empty content
        for i, section in enumerate(sections):
            if not section.content.strip():
                self.logger.warning(f"Empty section '{section.title}' in {pdf_path}")
                # Set minimal content
                sections[i].content = f"[Empty section: {section.title}]"
    
    def get_download_url(self, filename: str) -> str:
        # First try to find the exact filename in our dictionary
        if filename in self.download_urls:
            return self.download_urls[filename]
        
        # If not found, try to find a match ignoring case
        for key, url in self.download_urls.items():
            if filename.lower() == key.lower():
                return url
        
        # If still not found, try to extract document ID from filename and generate URL
        doc_id = self.extract_document_id(filename)
        if doc_id:
            return f"https://ntrs.nasa.gov/api/citations/{doc_id}/downloads/{filename}?attachment=true"
        
        # Last resort: search for partial matches in the log-extracted URLs
        for key, url in self.download_urls.items():
            # Check if the filename is contained within the key or vice versa
            if filename in key or key in filename:
                return url
        
        # If all else fails, return a search URL
        self.logger.warning(f"Could not find or generate download URL for {filename}")
        safe_filename = filename.replace(' ', '%20').replace('.pdf', '')
        return f"https://ntrs.nasa.gov/search?q={safe_filename}"
    
    def extract_document_id(self, filename):
        # Try to extract numeric ID (common pattern)
        numeric_match = re.search(r'(\d{8,})', filename)
        if numeric_match:
            return numeric_match.group(1)
        
        # Try to extract NASA document ID format (e.g., NASA-TM-123456)
        nasa_id_match = re.search(r'(NASA-[A-Z]+-\d+)', filename, re.IGNORECASE)
        if nasa_id_match:
            return nasa_id_match.group(1)
        
        return None
    
    def extract_download_urls_from_log(self, log_file):
        """Extract download URLs from the log file"""
        download_urls = {}
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # Find all download patterns in the log
                download_patterns = re.findall(r'Downloading from: (https://ntrs\.nasa\.gov/api/citations/\d+/downloads/.*?)\?attachment=true\n.*?Downloaded \d+/\d+: (.*?\.pdf)', log_content)
                
                for url, filename in download_patterns:
                    download_urls[filename] = url + "?attachment=true"
                    
            return download_urls
        except Exception as e:
            self.logger.error(f"Error extracting download URLs: {str(e)}")
            return {}

if __name__ == "__main__":
    chunker = SkippedFilesChunker()
    chunker.process_pdfs() 