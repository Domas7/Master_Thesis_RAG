import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime
import json
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import numpy as np

# Set style for better visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

DATES_CACHE_FILE = 'lesson_dates_cache.json'

def extract_year_from_pdf_name(filename):
    # Extract year from NASA document IDs (e.g., 20140005338.pdf -> 2014)
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        year = int(match.group(0))
        if 1999 <= year <= 2025:  # Specific range for technical documents
            return year
    return None

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')  # Set a standard window size
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # Prevent detection
    chrome_options.add_argument('--enable-javascript')  # Ensure JavaScript is enabled
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])  # Prevent detection
    chrome_options.add_experimental_option('useAutomationExtension', False)  # Prevent detection
    chrome_options.binary_location = '/usr/bin/chromium-browser'  # Use Chromium
    
    service = Service('/usr/bin/chromedriver')  # Use installed ChromeDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set page load timeout
    driver.set_page_load_timeout(30)
    
    # Set script timeout
    driver.set_script_timeout(30)
    
    return driver

def extract_date_from_url(url, driver):
    try:
        # Extract lesson number from URL
        lesson_number = url.split('/')[-1]
        
        # Construct the proper URL
        url = f"https://llis.nasa.gov/lesson/{lesson_number}"
        
        print(f"\nFetching URL: {url}")
        
        # Load the page
        driver.get(url)
        
        # Wait for the page to be fully loaded
        wait = WebDriverWait(driver, 20)  # Increased timeout to 20 seconds
        
        def get_date():
            try:
                # Wait for dl-horizontal to be present and visible
                dl_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "dl-horizontal"))
                )
                wait.until(
                    EC.visibility_of_element_located((By.CLASS_NAME, "dl-horizontal"))
                )
                
                # Wait for dt elements to be present
                wait.until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "dt"))
                )
                
                # Get all dt and dd elements
                dt_elements = dl_element.find_elements(By.TAG_NAME, "dt")
                dd_elements = dl_element.find_elements(By.TAG_NAME, "dd")
                
                # Look for the Lesson Date
                for dt, dd in zip(dt_elements, dd_elements):
                    try:
                        dt_text = dt.text.strip()
                        if dt_text == "Lesson Date":
                            # Wait for the dd element to be populated
                            wait.until(lambda d: dd.text.strip() != "")
                            dd_text = dd.text.strip()
                            print(f"Found date: {dd_text}")
                            return dd_text
                    except Exception as e:
                        print(f"Error processing element: {str(e)}")
                        continue
                return None
            except Exception as e:
                print(f"Error in get_date: {str(e)}")
                return None
        
        # Try multiple times to get the date
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                date_str = get_date()
                if date_str:
                    try:
                        year = datetime.strptime(date_str, '%Y-%m-%d').year
                        print(f"Successfully extracted year: {year}")
                        return url, year
                    except ValueError:
                        print(f"Failed to parse date: {date_str}")
                
                if attempt < max_attempts - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    driver.refresh()
                    time.sleep(2)  # Wait before retry
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_attempts - 1:
                    driver.refresh()
                    time.sleep(2)  # Wait before retry
        
        print(f"No lesson date found for {url} after {max_attempts} attempts")
        return url, None
        
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return url, None

def process_urls_batch(urls, existing_cache=None):
    if existing_cache is None:
        existing_cache = {}
    
    # Filter out URLs that are already in cache
    urls_to_scrape = [url for url in urls if url not in existing_cache]
    
    if not urls_to_scrape:
        return existing_cache
    
    # Initialize the webdriver
    driver = setup_driver()
    
    try:
        # For testing, let's just process the first few URLs
        test_urls = urls_to_scrape[:5]
        print(f"\nTesting with first {len(test_urls)} URLs:")
        for url in test_urls:
            print(f"\nProcessing URL: {url}")
            result = extract_date_from_url(url, driver)
            if result[1]:
                existing_cache[result[0]] = result[1]
                print(f"Successfully extracted year: {result[1]}")
            else:
                print("Failed to extract year")
        
        print("\nFirst batch results:")
        print(json.dumps(existing_cache, indent=2))
        
        # Ask user if they want to continue with all URLs
        response = input("\nContinue with all URLs? (y/n): ")
        if response.lower() != 'y':
            return existing_cache
        
        # Process all remaining URLs
        for url in tqdm(urls_to_scrape[5:], desc="Extracting dates"):
            result = extract_date_from_url(url, driver)
            if result[1] is not None:
                existing_cache[result[0]] = result[1]
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.1)
    
    finally:
        # Make sure to close the browser
        driver.quit()
    
    return existing_cache

def analyze_dataset():
    # 1. Read and analyze lessons learned data first
    print("\nAnalyzing lessons learned data...")
    lessons_df = pd.read_csv('NASA_Lessons_Learned/nasa_lessons_learned_centers_1.csv')
    
    # Load cached dates
    print("\nLoading cached dates...")
    url_dates = {}
    if os.path.exists(DATES_CACHE_FILE):
        with open(DATES_CACHE_FILE, 'r') as f:
            url_dates = json.load(f)
    
    # Filter lessons learned years to be between 1991 and 2018
    lesson_years = []
    for year in url_dates.values():
        if year and 1991 <= year <= 2018:
            lesson_years.append(year)
    
    # 2. Analyze PDF files for timeline
    print("\nAnalyzing PDF files...")
    pdf_dir = "NTRS_PDFS_CONFERENCE_GLOBAL"
    pdf_years = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            year = extract_year_from_pdf_name(filename)
            if year:
                pdf_years.append(year)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = plt.figure(figsize=(15, 8))  # Adjusted figure size to be more appropriate for horizontal layout
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # Changed to 1 row, 2 columns
    
    # Timeline Distributions
    ax1 = fig.add_subplot(gs[0])  # Changed from gs[0] to gs[0]
    if pdf_years:
        sns.histplot(data=pdf_years, bins=len(set(pdf_years)), ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Publication Dates of NASA Technical Documents', fontsize=18, pad=20)  # Slightly reduced font size
        ax1.set_xlabel('Year', fontsize=15)
        ax1.set_ylabel('Number of Documents', fontsize=17)
        ax1.set_xlim(1998, 2026)
        ax1.set_xticks(range(1999, 2026, 2))
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Lessons Learned Timeline Distribution
    ax2 = fig.add_subplot(gs[1])  # Changed from gs[1] to gs[1]
    if lesson_years:
        sns.histplot(data=lesson_years, bins=len(set(lesson_years)), ax=ax2, color='lightgreen', edgecolor='black')
        ax2.set_title('Publication Dates of NASA Lessons Learned Documents', fontsize=18, pad=20)
        ax2.set_xlabel('Year', fontsize=15)
        ax2.set_ylabel('Number of Documents', fontsize=17)
        ax2.set_xlim(1990, 2019)  # Adjusted range for lessons learned
        ax2.set_xticks(range(1991, 2019, 2))
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"\nTechnical Documents:")
    print(f"Total number of documents with valid years: {len(pdf_years)}")
    if pdf_years:
        print(f"Year range: {min(pdf_years)} - {max(pdf_years)}")
        
        # Print year distribution
        year_counts = pd.Series(pdf_years).value_counts().sort_index()
        print("\nDocuments per year:")
        for year, count in year_counts.items():
            print(f"{year}: {count} documents")
    
    print(f"\nLessons Learned:")
    if lesson_years:
        print(f"Total number of lessons with valid dates: {len(lesson_years)}")
        print(f"Year range: {min(lesson_years)} - {max(lesson_years)}")
        
        # Print year distribution
        year_counts = pd.Series(lesson_years).value_counts().sort_index()
        print("\nLessons per year:")
        for year, count in year_counts.items():
            print(f"{year}: {count} lessons")

    print("\nAnalyzing PDF file sizes...")
    analyze_pdf_sizes(pdf_dir)


import os
from pathlib import Path

def analyze_pdf_sizes(pdf_dir):
    # Define more granular size ranges in bytes (0.2MB increments up to 2MB, then larger ranges)
    size_ranges = [
        (0, 0.2*1024*1024),          # 0-0.2MB
        (0.2*1024*1024, 0.4*1024*1024),  # 0.2-0.4MB
        (0.4*1024*1024, 0.6*1024*1024),  # 0.4-0.6MB
        (0.6*1024*1024, 0.8*1024*1024),  # 0.6-0.8MB
        (0.8*1024*1024, 1*1024*1024),    # 0.8-1MB
        (1*1024*1024, 1.2*1024*1024),    # 1-1.2MB
        (1.2*1024*1024, 1.4*1024*1024),  # 1.2-1.4MB
        (1.4*1024*1024, 1.6*1024*1024),  # 1.4-1.6MB
        (1.6*1024*1024, 1.8*1024*1024),  # 1.6-1.8MB
        (1.8*1024*1024, 2*1024*1024),    # 1.8-2MB
        (2*1024*1024, 3*1024*1024),      # 2-3MB
        (3*1024*1024, 4*1024*1024),      # 3-4MB
        (4*1024*1024, 5*1024*1024),      # 4-5MB
        (5*1024*1024, 7*1024*1024),      # 5-7MB
        (7*1024*1024, 10*1024*1024),     # 7-10MB
        (10*1024*1024, float('inf'))      # 10MB+
    ]
    
    # Initialize counters for each range
    size_counts = {f"{size[0]/(1024*1024):.1f}-{size[1]/(1024*1024):.1f}MB": 0 for size in size_ranges}
    
    # Count files in each range
    for file in Path(pdf_dir).glob('*.pdf'):
        size = file.stat().st_size
        for (min_size, max_size) in size_ranges:
            if min_size <= size < max_size:
                range_key = f"{min_size/(1024*1024):.1f}-{max_size/(1024*1024):.1f}MB"
                size_counts[range_key] += 1
                break
    
    # Create visualization with adjusted figure size for more columns
    plt.figure(figsize=(15, 8))
    bars = plt.bar(size_counts.keys(), size_counts.values(), color='lightcoral', edgecolor='black')
    
    plt.title('Distribution of PDF File Sizes', fontsize=18, pad=20)
    plt.xlabel('File Size Range', fontsize=15)
    plt.ylabel('Number of Files', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pdf_size_distribution.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    total_size = sum(file.stat().st_size for file in Path(pdf_dir).glob('*.pdf'))
    num_files = sum(size_counts.values())
    
    print("\nPDF Size Distribution Summary:")
    print(f"Total number of PDFs: {num_files}")
    print(f"Total size of all PDFs: {total_size/(1024*1024*1024):.2f} GB")
    print(f"Average PDF size: {(total_size/num_files)/(1024*1024):.2f} MB")
    print("\nFiles in each size range:")
    for range_name, count in size_counts.items():
        print(f"{range_name}: {count} files")

def analyze_word_counts():
    # Create statistics directory if it doesn't exist
    os.makedirs('statistics', exist_ok=True)
    
    # Load multiple JSON files
    json_files = [
        '../reprocessed_section_chunks/reprocessed_section_chunks_final.json',
        '../reprocessed_section_chunks_2/reprocessed_section_chunks_2_final.json',
        '../reprocessed_section_chunks_3/reprocessed_section_chunks_3_final.json'
    ]
    
    # Store word counts for each file
    word_counts_by_file = {}
    
    # Process each file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Simplify the file key name for better display
                file_key = f"Chunk Set {len(word_counts_by_file) + 1}"
                
                # Extract word counts from each chunk
                word_counts = [
                    chunk.get('metadata', {}).get('statistics', {}).get('word_count', 0) 
                    for chunk in data
                ]
                word_counts_by_file[file_key] = word_counts
                print(f"Processed {file_key}: {len(word_counts)} chunks")
                
        except FileNotFoundError:
            print(f"Warning: File {json_file} not found, skipping.")
        except json.JSONDecodeError:
            print(f"Warning: File {json_file} contains invalid JSON, skipping.")
    
    # Create visualization
    plt.figure(figsize=(15, 6))
    
    # Create boxplot for word count distributions
    plt.subplot(1, 2, 1)
    box_data = [counts for counts in word_counts_by_file.values()]
    plt.boxplot(box_data, labels=word_counts_by_file.keys())
    plt.title('Word Count Distribution by Chunk Set')
    plt.ylabel('Number of Words per Chunk')
    plt.grid(True, alpha=0.3)
    
    # Create violin plot for more detailed distribution view
    plt.subplot(1, 2, 2)
    violin_parts = plt.violinplot(box_data)
    plt.xticks(range(1, len(word_counts_by_file) + 1), word_counts_by_file.keys())
    plt.title('Word Count Density Distribution')
    plt.ylabel('Number of Words per Chunk')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    for i, (file_name, counts) in enumerate(word_counts_by_file.items()):
        stats_text = (f"{file_name}:\n"
                     f"Mean: {np.mean(counts):.0f}\n"
                     f"Median: {np.median(counts):.0f}\n"
                     f"Total chunks: {len(counts)}")
        plt.text(0.02, 0.98 - (i * 0.15), stats_text,
                transform=plt.gcf().transFigure,
                fontsize=9,
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('statistics/word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print overall statistics
    print("\nOverall Statistics:")
    for file_name, counts in word_counts_by_file.items():
        print(f"\n{file_name}:")
        print(f"Total chunks: {len(counts)}")
        print(f"Average words per chunk: {np.mean(counts):.1f}")
        print(f"Median words per chunk: {np.median(counts):.1f}")
        print(f"Min words: {min(counts)}")
        print(f"Max words: {max(counts)}")

def analyze_total_word_counts():
    # Create statistics directory if it doesn't exist
    os.makedirs('statistics', exist_ok=True)
    
    # Define word count ranges with 500-word increments up to 5000, then 1000-word increments
    word_ranges = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
        (2000, 2500),
        (2500, 3000),
        (3000, 3500),
        (3500, 4000),
        (4000, 4500),
        (4500, 5000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
        (8000, 9000),
        (9000, 10000),
        (10000, float('inf'))
    ]
    
    # Load multiple JSON files
    json_files = [
        '../reprocessed_section_chunks/reprocessed_section_chunks_final.json',
        '../reprocessed_section_chunks_2/reprocessed_section_chunks_2_final.json',
        '../reprocessed_section_chunks_3/reprocessed_section_chunks_3_final.json'
    ]
    
    # Store word counts per PDF file
    pdf_word_counts = {}
    
    # Process each file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for chunk in data:
                    # Extract the PDF filename (everything before .pdf)
                    pdf_name = chunk['metadata']['file_name'].split('.pdf')[0]
                    word_count = chunk['metadata']['statistics']['word_count']
                    
                    # Add to existing count or create new entry
                    if pdf_name in pdf_word_counts:
                        pdf_word_counts[pdf_name] += word_count
                    else:
                        pdf_word_counts[pdf_name] = word_count
                
        except FileNotFoundError:
            print(f"Warning: File {json_file} not found, skipping.")
        except json.JSONDecodeError:
            print(f"Warning: File {json_file} contains invalid JSON, skipping.")
    
    # Count files in each range
    range_counts = {f"{range[0]}-{range[1] if range[1] != float('inf') else '+'} words": 0 
                   for range in word_ranges}
    
    # Categorize files into ranges
    for total_words in pdf_word_counts.values():
        for min_words, max_words in word_ranges:
            if min_words <= total_words < max_words:
                range_key = f"{min_words}-{max_words if max_words != float('inf') else '+'} words"
                range_counts[range_key] += 1
                break
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range_counts.keys(), range_counts.values(), color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Total Words per PDF File', fontsize=14)
    plt.xlabel('Word Count Range', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only show label if there are files in this range
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('statistics/total_words_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed statistics
    print("\nTotal Words Per PDF File:")
    for pdf_name, total_words in sorted(pdf_word_counts.items()):
        print(f"{pdf_name}: {total_words:,} words")
    
    print("\nFiles in Each Range:")
    for range_name, count in range_counts.items():
        print(f"{range_name}: {count} files")

if __name__ == "__main__":
    analyze_dataset()
    analyze_word_counts()
    analyze_total_word_counts() 