import os
import json
import logging
import time
import datetime
import re
from typing import List, Dict, Any, Tuple, Set
import pymupdf4llm
from dataclasses import dataclass
from tqdm import tqdm
import multiprocessing
from functools import partial
import signal
from contextlib import contextmanager
import copy

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
class Section:
    """Class for keeping track of a section in a document."""
    title: str
    content: str
    page_number: int
    section_level: int
    section_number: int
    total_sections: int

class SingleSectionReprocessor:
    """Class to reprocess PDFs that were incorrectly processed as single sections."""
    
    def __init__(self):
        self.input_dir = "RAG/NTRS_PDFS_CONFERENCE_GLOBAL"
        self.output_dir = "reprocessed_section_chunks_3"
        self.temp_dir = "temp_reprocessed"
        self.log_file = "reprocessed_files.log"
        self.timeout_seconds = 120
        
        # Create output and temp directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("SingleSectionReprocessor")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Load the list of files to reprocess
        self.files_to_reprocess = self.get_files_to_reprocess()
        self.logger.info(f"Found {len(self.files_to_reprocess)} files to reprocess")
        
        # Download URLs
        self.download_urls = self.extract_download_urls_from_log("RAG/NASA_Technical_Server/nasa_pdf_processing_DONE.log")
        
    def get_files_to_reprocess(self) -> List[str]:
        """Get the list of files that need to be reprocessed."""
        # Initialize the set of files to reprocess
        single_section_files = set()
        
        # First, try to find files with total_sections = 1 in the chunks file
        try:
            with open("RAG/SPACECRAFT22_section_chunks_CHECKTHIS.json", 'r') as f:
                chunks = json.load(f)
                
                # Only add files where total_sections = 1
                for chunk in chunks:
                    if chunk.get("total_sections") == 1 and "file_name" in chunk.get("metadata", {}):
                        single_section_files.add(chunk["metadata"]["file_name"])
                
                self.logger.info(f"Found {len(single_section_files)} files with total_sections = 1 in chunks file")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error reading chunks file: {str(e)}")
        
        # If we didn't find any single-section files in the chunks, or want to add more,
        # we can also get them from the skipped_stats file
        if len(single_section_files) == 0:
            try:
                with open("RAG/chunk_statistics.json", 'r') as f:
                    skipped_stats = json.load(f)
                    
                    if "still_skipped_file_list" in skipped_stats:
                        # Only add files that aren't already in our set
                        for file_name in skipped_stats["still_skipped_file_list"]:
                            if file_name.endswith(".pdf"):
                                single_section_files.add(file_name)
                        
                        self.logger.info(f"Added {len(single_section_files)} files from skipped_stats")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(f"Error reading skipped_stats file: {str(e)}")
        
        self.logger.info(f"Total files to reprocess: {len(single_section_files)}")
        return list(single_section_files)
    
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
    
    def get_download_url(self, filename: str) -> str:
        """Get download URL for a file, with multiple fallback methods"""
        # First try direct lookup
        if filename in self.download_urls:
            return self.download_urls[filename]
        
        # Try to extract NASA ID from filename
        nasa_id_match = re.match(r'^(\d+)\.pdf$', filename)
        if nasa_id_match:
            nasa_id = nasa_id_match.group(1)
            return f"https://ntrs.nasa.gov/api/citations/{nasa_id}/downloads/{filename}?attachment=true"
        
        # Another pattern: 20XXXXXXXX.pdf
        nasa_id_match = re.match(r'^(20\d{8})\.pdf$', filename)
        if nasa_id_match:
            nasa_id = nasa_id_match.group(1)
            return f"https://ntrs.nasa.gov/api/citations/{nasa_id}/downloads/{filename}?attachment=true"
        
        return None
    
    def extract_sections_from_markdown(self, markdown_text: str, pdf_path: str) -> List[Section]:
        """Extract sections from markdown text ensuring ALL content is preserved"""
        sections = []
        page_number = 1
        
        try:
            # STEP 1: Thoroughly preprocess to remove all page and column break artifacts
            # This is the critical step to prevent creating chunks at column breaks
            
            # First, normalize line endings and remove page break markers
            lines = markdown_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip page break markers
                if ((line.startswith('---') and 'Page' in line) or 
                    line == '-----' or 
                    re.match(r'^-+$', line)):
                    try:
                        if 'Page' in line:
                            page_number = int(line.split('Page')[1].strip().split()[0])
                    except:
                        pass
                    continue
                
                # Keep non-empty lines
                if line.strip():
                    cleaned_lines.append(line)
            
            # STEP 2: Handle column breaks by analyzing text flow
            # This is similar to how your main chunking script works
            processed_lines = []
            i = 0
            
            while i < len(cleaned_lines):
                current_line = cleaned_lines[i].strip()
                
                # Skip empty lines
                if not current_line:
                    i += 1
                    continue
                
                # Check if this line might continue in the next line (column break)
                if i < len(cleaned_lines) - 1:
                    next_line = cleaned_lines[i+1].strip()
                    
                    # Conditions that suggest this is a column break:
                    # 1. Current line doesn't end with punctuation
                    # 2. Next line starts with lowercase
                    # 3. Current line ends with hyphen (hyphenated word)
                    # 4. Current line is very short (likely cut off by column)
                    
                    is_column_break = False
                    
                    # Check for sentence continuation
                    if (next_line and 
                        (not current_line[-1] in '.,:;!?"\')]}') and 
                        (next_line[0].islower() or next_line[0] in '([{"\'')):
                        is_column_break = True
                    
                    # Check for hyphenated word
                    elif current_line.endswith('-') and next_line and next_line[0].islower():
                        is_column_break = True
                        current_line = current_line[:-1]  # Remove hyphen
                    
                    # Check for short line (likely column break)
                    elif (len(current_line) < 40 and next_line and 
                          not re.match(r'^\s*\*\*.*\*\*\s*$', current_line) and  # Not a bold header
                          not re.match(r'^\d+\.', current_line) and  # Not a numbered list
                          not re.match(r'^[A-Z][A-Z\s]+$', current_line)):  # Not an all-caps header
                        is_column_break = True
                    
                    # If this is a column break, join with next line
                    if is_column_break:
                        processed_lines.append(current_line + ' ' + next_line)
                        i += 2  # Skip both lines
                        continue
                
                # Regular line, keep it
                processed_lines.append(current_line)
                i += 1
            
            # Join the processed lines
            processed_markdown = '\n'.join(processed_lines)
            
            # STEP 3: Clean up the text further
            # Replace multiple spaces with single space
            processed_markdown = re.sub(r' {2,}', ' ', processed_markdown)
            
            # Normalize newlines
            processed_markdown = re.sub(r'\n{4,}', '\n\n\n', processed_markdown)
            
            # STEP 4: Find section breaks based on headers and triple newlines
            section_breaks = []
            
            # Find all bold headers with newlines
            bold_headers_with_newlines = re.finditer(r'\n\s*\*\*([^*\n]+)\*\*\s*\n', processed_markdown)
            for match in bold_headers_with_newlines:
                header_text = match.group(1).strip()
                # Skip if this is a figure or table header
                if not re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                               header_text.lower()):
                    section_breaks.append(match.start() + 1)
            
            # Find bold headers without newlines (adjacent bold sections)
            # This handles cases like "**Introduction** **Results and Discussion**"
            bold_headers_adjacent = re.finditer(r'\*\*([^*\n]+)\*\*\s+\*\*', processed_markdown)
            for match in bold_headers_adjacent:
                header_text = match.group(1).strip()
                # Skip if this is a figure or table header
                if not re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                               header_text.lower()):
                    # Add a newline before the second bold header to create a section break
                    pos = match.end() - 2  # Position right before the second **
                    section_breaks.append(pos)
            
            # Find all-caps headers
            caps_headers = re.finditer(r'\n\s*([A-Z][A-Z\s]{2,}[A-Z])\s*\n', processed_markdown)
            for match in caps_headers:
                header_text = match.group(1).strip()
                # Skip if this is a figure or table header
                if not re.match(r'^(FIG|FIGURE|TAB|TABLE|CHART|GRAPH|IMAGE|PICTURE|DIAGRAM)\.?\s*\d*', 
                               header_text):
                    section_breaks.append(match.start() + 1)
            
            # Find numbered section headers
            numbered_headers = re.finditer(r'\n\s*(\d+\.?\s+[A-Z][a-zA-Z\s]+)\s*\n', processed_markdown)
            for match in numbered_headers:
                header_text = match.group(1).strip()
                # Skip if this is a figure or table header
                if not re.match(r'^\d+\.?\s+(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                               header_text.lower()):
                    section_breaks.append(match.start() + 1)
            
            # Find triple newlines as fallback section breaks
            triple_newlines = re.finditer(r'\n\n\n', processed_markdown)
            for match in triple_newlines:
                # Check if the text after the triple newline is a figure/table caption
                next_text = processed_markdown[match.end():match.end()+50].strip()
                if not re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                               next_text.lower()):
                    section_breaks.append(match.start())
            
            # Sort and deduplicate breaks
            section_breaks = sorted(set(section_breaks))
            
            # If we found section breaks, use them to split the text
            potential_sections = []
            if section_breaks:
                for i in range(len(section_breaks)):
                    start = section_breaks[i]
                    end = section_breaks[i+1] if i < len(section_breaks)-1 else len(processed_markdown)
                    section_text = processed_markdown[start:end].strip()
                    if section_text:
                        potential_sections.append(section_text)
            
                # Add the content before the first break if it exists
                if section_breaks[0] > 0:
                    first_section = processed_markdown[:section_breaks[0]].strip()
                    if first_section:
                        potential_sections.insert(0, first_section)
            else:
                # If no section breaks found, use the whole document
                potential_sections = [processed_markdown.strip()]
            
            # STEP 5: Process each potential section, handling adjacent bold headers
            for i, section_text in enumerate(potential_sections):
                # Check if this section contains adjacent bold headers
                bold_headers = re.findall(r'\*\*([^*\n]+)\*\*', section_text)
                
                # Filter out figure/table headers
                real_headers = []
                for header in bold_headers:
                    if not re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                                   header.lower()):
                        real_headers.append(header)
                
                # If we have multiple non-figure/table bold headers in the same section, split them
                if len(real_headers) > 1:
                    # Try to split at the bold headers
                    sub_sections = re.split(r'(\*\*[^*\n]+\*\*)', section_text)
                    
                    # Group header with its content
                    j = 0
                    while j < len(sub_sections) - 1:
                        if re.match(r'\*\*[^*\n]+\*\*', sub_sections[j]):
                            header_text = re.sub(r'\*\*|\*', '', sub_sections[j]).strip()
                            
                            # Skip figure/table headers
                            if re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                                       header_text.lower()):
                                # This is a figure/table, append to previous section if exists
                                if sections:
                                    sections[-1].content += "\n" + sub_sections[j]
                                    if j + 1 < len(sub_sections):
                                        sections[-1].content += sub_sections[j+1]
                                else:
                                    # No previous section, create a new one with default title
                                    title = f"Section {len(sections) + 1}"
                                    content = sub_sections[j]
                                    if j + 1 < len(sub_sections):
                                        content += sub_sections[j+1]
                                    
                                    sections.append(Section(
                                        title=title,
                                        content=content.strip(),
                                        page_number=page_number,
                                        section_level=1,
                                        section_number=len(sections) + 1,
                                        total_sections=0  # Will update later
                                    ))
                                j += 2
                            else:
                                # This is a real header, combine with the next element which is content
                                if j + 1 < len(sub_sections):
                                    title = header_text
                                    content = sub_sections[j+1].strip()
                                    
                                    # Create a section
                                    if content:  # Only create if there's content
                                        sections.append(Section(
                                            title=title,
                                            content=content,
                                            page_number=page_number,
                                            section_level=1,
                                            section_number=len(sections) + 1,
                                            total_sections=0  # Will update later
                                        ))
                                    j += 2
                                else:
                                    j += 1
                        else:
                            # This is content without a header, add it to the previous section or create a new one
                            if sub_sections[j].strip():
                                if sections and j > 0:
                                    # Append to the previous section
                                    sections[-1].content += "\n" + sub_sections[j].strip()
                                else:
                                    # Create a new section with a default title
                                    sections.append(Section(
                                        title=f"Section {len(sections) + 1}",
                                        content=sub_sections[j].strip(),
                                        page_number=page_number,
                                        section_level=1,
                                        section_number=len(sections) + 1,
                                        total_sections=0  # Will update later
                                    ))
                            j += 1
                else:
                    # Process as a normal section
                    lines = section_text.split('\n')
                    title = None
                    content_start = 0
                    
                    # Look for a header in the first few lines
                    for j in range(min(3, len(lines))):
                        line = lines[j].strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Check if this line or the previous line starts with figure/table indicators
                        is_figure_caption = False
                        if re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', line.lower()):
                            is_figure_caption = True
                        
                        # Also check if this is a continuation of a figure caption from previous lines
                        if j > 0:
                            prev_line = lines[j-1].strip().lower()
                            if re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', prev_line):
                                is_figure_caption = True
                        
                        # Skip figure captions
                        if is_figure_caption:
                            continue
                        
                        # Check if this line looks like a header
                        if (re.match(r'^\s*\*\*.*\*\*\s*$', line) or  # Bold text
                            re.match(r'^[A-Z][A-Z\s]{2,}[A-Z]$', line) or  # ALL CAPS
                            re.match(r'^\d+\.?\s+[A-Z]', line) or  # Numbered section
                            len(line) < 80):  # Not too long
                            
                            # Skip if this is a figure or table header
                            clean_line = re.sub(r'\*\*|\*', '', line).strip()
                            if re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                                       clean_line.lower()):
                                continue
                            
                            title = line
                            # Clean up the title
                            title = re.sub(r'\*\*|\*', '', title)  # Remove bold markers
                            title = re.sub(r'^\s*\d+\.?\s+', '', title)  # Remove section numbers
                            content_start = j + 1
                            break
                    
                    # If no title found, use the first line
                    if not title and lines:
                        first_line = lines[0].strip()
                        # Skip if the first line is a figure/table caption
                        if not re.match(r'^(fig|figure|tab|table|chart|graph|image|picture|diagram)\.?\s*\d*', 
                                       first_line.lower()):
                            title = first_line[:50]
                            if len(first_line) > 50:
                                title += "..."
                            content_start = 1
                    
                    # If still no title, use a default
                    if not title:
                        title = f"Section {len(sections) + 1}"
                        content_start = 0
                    
                    # Get the content (either the whole section or everything after the title)
                    if content_start < len(lines):
                        content = '\n'.join(lines[content_start:])
                        # If content is very short, include the title line as well
                        if len(content.strip()) < 50 and content_start > 0:
                            content = section_text
                    else:
                        content = section_text
                    
                    # Create the section
                    sections.append(Section(
                        title=title,
                        content=content.strip(),
                        page_number=page_number,
                        section_level=1,
                        section_number=len(sections) + 1,
                        total_sections=0  # Will update later
                    ))
            
            # If no sections were created, create one with all content
            if not sections:
                sections.append(Section(
                    title=f"Document: {os.path.basename(pdf_path)}",
                    content=processed_markdown,
                    page_number=1,
                    section_level=1,
                    section_number=1,
                    total_sections=1
                ))
            
            # Update total sections and section numbers
            total_sections = len(sections)
            for i, section in enumerate(sections):
                section.section_number = i+1
                section.total_sections = total_sections
            
            return sections
        
        except Exception as e:
            self.logger.error(f"Error extracting sections from {pdf_path}: {str(e)}")
            # Create an error section with all content
            error_section = Section(
                title=f"Error processing {os.path.basename(pdf_path)}",
                content=markdown_text,
                page_number=1,
                section_level=1,
                section_number=1,
                total_sections=1
            )
            return [error_section]
    
    def post_process_sections(self, sections: List[Section]) -> List[Section]:
        """Post-process sections to clean up and merge if needed"""
        if not sections:
            return []
        
        # Merge very short sections with the next section
        merged_sections = []
        i = 0
        while i < len(sections):
            current = sections[i]
            
            # If this is a very short section and not the last one, merge with next
            if (len(current.content.split()) < 50 and 
                i < len(sections) - 1 and 
                not re.search(r'(abstract|introduction|conclusion|references|acknowledgements)', 
                             current.title, re.IGNORECASE)):
                next_section = sections[i+1]
                next_section.content = f"{current.title}\n\n{current.content}\n\n{next_section.content}"
                i += 1
            else:
                merged_sections.append(current)
                i += 1
        
        # Update section numbers and total
        total_sections = len(merged_sections)
        for i, section in enumerate(merged_sections):
            section.section_number = i + 1
            section.total_sections = total_sections
        
        return merged_sections
    
    def verify_sections(self, sections: List[Section], pdf_path: str) -> None:
        """Verify that sections are valid and fix if needed"""
        # If no sections were found, create a default section
        if not sections:
            self.logger.warning(f"No sections found in {pdf_path}, creating default section")
            sections.append(Section(
                title=f"Document: {os.path.basename(pdf_path)}",
                content="No content could be extracted from this document.",
                page_number=1,
                section_level=1,
                section_number=1,
                total_sections=1
            ))
            return
        
        # Check for empty sections
        for i, section in enumerate(sections):
            if not section.content.strip():
                self.logger.warning(f"Empty section '{section.title}' in {pdf_path}")
                # Set minimal content
                sections[i].content = f"[Empty section: {section.title}]"
    
    def process_single_pdf(self, pdf_file: str) -> Tuple[List[Dict], List[str], Dict]:
        """Process a single PDF file and extract sections"""
        pdf_path = os.path.join(self.input_dir, pdf_file)
        stats = {
            "file_name": pdf_file,
            "processing_time_seconds": 0,
            "sections_found": 0,
            "word_count": 0,
            "character_count": 0
        }
        
        error_info = None
        file_chunks = []
        
        start_time = time.time()
        
        try:
            # Extract text from PDF with timeout
            with time_limit(self.timeout_seconds):
                markdown_text = pymupdf4llm.to_markdown(pdf_path)
            
            # Extract sections
            sections = self.extract_sections_from_markdown(markdown_text, pdf_path)
            
            # Post-process sections
            sections = self.post_process_sections(sections)
            
            # Verify sections
            self.verify_sections(sections, pdf_path)
            
            # Calculate statistics
            total_words = 0
            total_chars = 0
            
            # Get download URL
            download_url = self.get_download_url(pdf_file)
            
            # Convert sections to chunks
            for section in sections:
                section_words = len(section.content.split())
                section_chars = len(section.content)
                
                total_words += section_words
                total_chars += section_chars
                
                # Create chunk
                chunk = {
                    "chunk_id": f"{pdf_file}_{section.section_number - 1}",
                    "title": section.title,
                    "content": section.content,
                    "page_number": section.page_number,
                    "section_level": section.section_level,
                    "section_number": section.section_number,
                    "total_sections": section.total_sections,
                    "metadata": {
                        "file_name": pdf_file,
                        "file_path": pdf_path,
                        "download_url": download_url,
                        "statistics": {
                            "word_count": section_words,
                            "character_count": section_chars,
                            "average_word_length": section_chars / section_words if section_words > 0 else 0,
                            "processing_time_seconds": time.time() - start_time
                        }
                    }
                }
                file_chunks.append(chunk)
            
            # Update stats
            stats["processing_time_seconds"] = time.time() - start_time
            stats["sections_found"] = len(sections)
            stats["word_count"] = total_words
            stats["character_count"] = total_chars
            
            self.logger.info(f"Successfully processed {pdf_file}: found {len(sections)} sections, {total_words} words")
            
        except TimeoutException:
            error_info = f"Timeout processing {pdf_file} after {self.timeout_seconds} seconds"
            self.logger.error(error_info)
        except Exception as e:
            error_info = f"Error processing {pdf_file}: {str(e)}"
            self.logger.error(error_info)
            
            # Write error details to a file
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_error.txt"), 'w') as f:
                f.write(f"Error: {str(e)}")
        
        return file_chunks, [error_info] if error_info else [], stats
    
    def process_pdfs(self):
        """Process all PDFs in the list"""
        start_time = time.time()
        
        # Use multiprocessing to speed up processing
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        self.logger.info(f"Using {num_cores} cores for processing")
        
        # Process files in batches to avoid memory issues
        batch_size = 100
        all_chunks = []
        all_errors = []
        all_stats = []
        
        for i in range(0, len(self.files_to_reprocess), batch_size):
            batch = self.files_to_reprocess[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.files_to_reprocess) + batch_size - 1)//batch_size}")
            
            with multiprocessing.Pool(num_cores) as pool:
                results = list(tqdm(
                    pool.imap(self.process_single_pdf, batch),
                    total=len(batch),
                    desc="Processing PDFs"
                ))
            
            # Collect results
            for chunks, errors, stats in results:
                all_chunks.extend(chunks)
                all_errors.extend(errors)
                all_stats.append(stats)
            
            # Save intermediate results
            self.save_chunks(all_chunks, f"intermediate_batch_{i//batch_size + 1}")
        
        # Save final results
        self.save_chunks(all_chunks)
        self.save_statistics(all_stats, all_errors)
        
        end_time = time.time()
        self.logger.info(f"Finished processing {len(self.files_to_reprocess)} PDFs in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Total chunks: {len(all_chunks)}")
        self.logger.info(f"Total errors: {len(all_errors)}")
    
    def save_chunks(self, chunks, suffix="final"):
        """Save chunks to a JSON file"""
        output_file = os.path.join(self.output_dir, f"reprocessed_section_chunks_3_{suffix}.json")
        with open(output_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        self.logger.info(f"Saved {len(chunks)} chunks to {output_file}")
    
    def save_statistics(self, stats, errors):
        """Save statistics to a JSON file"""
        # Calculate overall statistics
        total_pdfs = len(self.files_to_reprocess)
        total_pdfs_processed = len(stats)
        total_pdfs_with_errors = len(errors)
        
        # Calculate chunk statistics
        total_chunks = sum(stat["sections_found"] for stat in stats)
        avg_chunks_per_pdf = total_chunks / total_pdfs_processed if total_pdfs_processed > 0 else 0
        
        # Word and character statistics
        total_words = sum(stat["word_count"] for stat in stats)
        total_chars = sum(stat["character_count"] for stat in stats)
        avg_words_per_chunk = total_words / total_chunks if total_chunks > 0 else 0
        avg_chars_per_chunk = total_chars / total_chunks if total_chunks > 0 else 0
        
        # Min/max statistics
        min_words = min((stat["word_count"] for stat in stats), default=0)
        max_words = max((stat["word_count"] for stat in stats), default=0)
        min_chars = min((stat["character_count"] for stat in stats), default=0)
        max_chars = max((stat["character_count"] for stat in stats), default=0)
        
        # Create statistics object
        statistics = {
            "total_pdfs_attempted": total_pdfs,
            "total_pdfs_processed": total_pdfs_processed,
            "total_pdfs_with_errors": total_pdfs_with_errors,
            "error_list": errors,
            "total_chunks": total_chunks,
            "average_chunks_per_pdf": avg_chunks_per_pdf,
            "average_words_per_chunk": avg_words_per_chunk,
            "average_chars_per_chunk": avg_chars_per_chunk,
            "min_words": min_words,
            "max_words": max_words,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save statistics
        output_file = os.path.join(self.output_dir, "reprocessed_statistics_3.json")
        with open(output_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        self.logger.info(f"Saved statistics to {output_file}")

if __name__ == "__main__":
    reprocessor = SingleSectionReprocessor()
    reprocessor.process_pdfs() 