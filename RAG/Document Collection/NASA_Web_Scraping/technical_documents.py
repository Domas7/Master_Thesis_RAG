import os
import json
import pandas as pd
import fitz  # PyMuPDF
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import re
from dateutil.parser import parse as parse_date
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# THIS SCRIPT ONLY USES THE UPPER CLASS TO DOWNLOAD THE PDFs FROM NASA TECHNICAL REPORTS SERVER
# THE LOWER CLASS IS USED TO PROCESS THE PDFs AND EXTRACT THE TEXT FROM THEM BUT IT IS NOT CURRENTLY BEING USED

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PDFDownloader:
    """Class to download PDFs from NASA Technical Reports Server."""
    
    def __init__(self, base_url, output_dir="NTRS_PDFS_CONFERENCE_GLOBAL", max_docs=8000, start_from=0):
        """
        Initialize the downloader.
        
        Args:
            base_url: The base URL for NASA Technical Reports Server
            output_dir: Directory to save downloaded PDFs
            max_docs: Maximum number of documents to download
            start_from: Document number to start from (for resuming)
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.start_from = start_from
        self.logger = logging.getLogger(__name__)
        self.base_domain = "https://ntrs.nasa.gov"
        self.progress_file = "download_progress.json"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Configure Chrome options for browsing
        self.chrome_options = Options()
        # Run in non-headless mode to ensure JavaScript renders properly
        # self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        
        # Add crash recovery options
        self.chrome_options.add_argument("--disable-crash-reporter")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-in-process-stack-traces")
        self.chrome_options.add_argument("--disable-logging")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--log-level=3")
        self.chrome_options.add_argument("--silent")
    
    def load_progress(self):
        """Load download progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    return progress.get('documents_downloaded', 0), progress.get('page_num', 0)
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
        return self.start_from, self.start_from // 100
    
    def save_progress(self, documents_downloaded, page_num):
        """Save download progress to file."""
        try:
            progress = {
                'documents_downloaded': documents_downloaded,
                'page_num': page_num,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def download_file(self, url, filename):
        """Download a file using requests."""
        # Skip if file already exists
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            self.logger.info(f"File already exists: {filename}")
            return True
            
        try:
            # If URL is relative, make it absolute
            if url.startswith('/'):
                url = f"{self.base_domain}{url}"
                
            self.logger.info(f"Downloading from: {url}")
            
            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Check if it's a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type and not url.endswith('.pdf'):
                self.logger.warning(f"Downloaded file may not be a PDF. Content-Type: {content_type}")
            
            # Save the file
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was downloaded and is not empty
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return True
            else:
                self.logger.error(f"File download failed or file is empty: {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading file {url}: {e}")
            return False
    
    def download_pdfs(self):
        """Download PDFs from NASA Technical Reports Server."""
        # Load progress if resuming
        documents_downloaded, page_num = self.load_progress()
        self.logger.info(f"Starting download from document {documents_downloaded} (page {page_num+1})")
        
        # Count existing PDFs
        existing_pdfs = [f for f in os.listdir(self.output_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(existing_pdfs)} existing PDFs in the output directory")
        
        # Initialize driver with retry mechanism
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            driver = None
            try:
                driver = webdriver.Chrome(options=self.chrome_options)
                
                # Continue until we've downloaded enough documents or run out of pages
                while documents_downloaded < self.max_docs and page_num < 80:
                    # Use the exact URL format from the website
                    page_url = f"{self.base_url}&page=%7B%22size%22:100,%22from%22:{page_num*100}%7D&sort=%7B%22field%22:%22published%22,%22order%22:%22desc%22%7D"
                    self.logger.info(f"Processing page {page_num+1}, URL: {page_url}")
                    
                    driver.get(page_url)
                    # Wait for the page to load completely
                    time.sleep(10)
                    
                    # Extract download links directly from the page source
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, 'html.parser')
                    
                    # Find all download links that match the pattern
                    download_links = []
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag.get('href')
                        if '/downloads/' in href and '.pdf' in href:
                            # Get the title from the parent card
                            card = a_tag.find_parent('div', class_='search-result-card')
                            title = "Unknown"
                            if card:
                                title_tag = card.find('h3', class_='title')
                                if title_tag:
                                    title = title_tag.text.strip()
                            
                            download_links.append((title, href))
                    
                    self.logger.info(f"Found {len(download_links)} download links on page {page_num+1}")
                    
                   # Different approaches, first beautiful soup, then selenium
                    if not download_links:
                        self.logger.info("Trying direct Selenium approach to find download links")
                        
                        # Find all download buttons
                        download_buttons = driver.find_elements(By.CSS_SELECTOR, "a[title='Download Document']")
                        self.logger.info(f"Found {len(download_buttons)} download buttons with Selenium")
                        
                        for button in download_buttons:
                            try:
                                href = button.get_attribute('href')
                                if href and '/downloads/' in href and '.pdf' in href:
                                    # Try to get the title
                                    card = button.find_element(By.XPATH, "./ancestor::div[contains(@class, 'search-result-card')]")
                                    title = "Unknown"
                                    if card:
                                        try:
                                            title_elem = card.find_element(By.CSS_SELECTOR, "h3.title")
                                            title = title_elem.text.strip()
                                        except:
                                            pass
                                    
                                    download_links.append((title, href))
                            except Exception as e:
                                self.logger.warning(f"Error extracting link from button: {e}")
                    
                    # If still no links, save the page source for debugging
                    if not download_links:
                        with open(f"page_{page_num+1}_source.html", "w", encoding="utf-8") as f:
                            f.write(driver.page_source)
                        self.logger.warning(f"No download links found on page {page_num+1}. Page source saved for debugging.")
                    
                    # Download each PDF
                    for i, (title, href) in enumerate(download_links):
                        # Skip documents we've already processed if resuming
                        if documents_downloaded < self.start_from:
                            documents_downloaded += 1
                            continue
                            
                        if documents_downloaded >= self.max_docs:
                            break
                        
                        try:
                            # Extract filename from the download URL
                            if '/downloads/' in href:
                                # Extract the filename from the URL (part after /downloads/)
                                url_filename = href.split('/downloads/')[-1].split('?')[0]
                                # Remove URL encoding
                                url_filename = url_filename.replace('%20', ' ')
                                # Use this as the filename
                                filename = os.path.join(self.output_dir, url_filename)
                            else:
                                # Fallback to using title
                                safe_title = "".join(c if c.isalnum() else "_" for c in title)[:100]
                                filename = os.path.join(self.output_dir, f"{safe_title}.pdf")
                            
                            # Download the file
                            success = self.download_file(href, filename)
                            
                            if success:
                                documents_downloaded += 1
                                self.logger.info(f"Downloaded {documents_downloaded}/{self.max_docs}: {os.path.basename(filename)}")
                                
                                # Save progress every 10 documents
                                if documents_downloaded % 10 == 0:
                                    self.save_progress(documents_downloaded, page_num)
                            
                            # Don't overload the server
                            time.sleep(1)
                        
                        except Exception as e:
                            self.logger.error(f"Error processing document: {e}")
                            # Save progress in case of error
                            self.save_progress(documents_downloaded, page_num)
                    
                    page_num += 1
                    
                    # Save progress after each page
                    self.save_progress(documents_downloaded, page_num)
                    
                    # Add a delay between pages to avoid overloading the server
                    time.sleep(3)
                
                # If we got here without errors, break the retry loop
                break
                
            except Exception as e:
                self.logger.error(f"Error during download process: {e}")
                retry_count += 1
                self.logger.info(f"Retrying ({retry_count}/{max_retries})...")
                # Save progress before retrying
                self.save_progress(documents_downloaded, page_num)
                
                # Wait before retrying
                time.sleep(10)
            
            finally:
                # Always close the driver
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
        
        self.logger.info(f"Download complete. Downloaded {documents_downloaded} documents.")
        
        # Check if any PDFs were actually downloaded
        pdf_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files in the output directory")
        
        return len(pdf_files)
    
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("nasa_pdf_processing.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # NASA Technical Reports Server URL - use the exact URL you provided
    nasa_url = "https://ntrs.nasa.gov/search?stiTypeDetails=Conference%20Paper"
    
    # Get the starting document number from command line or use default
    import sys
    start_from = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
    
    # Step 1: Download PDFs to NTRS_PDFS folder, resuming from where we left off
    PDFDownloader(nasa_url, output_dir="NTRS_PDFS_CONFERENCE_GLOBAL", max_docs=10000, start_from=start_from)
    

if __name__ == "__main__":
    main()