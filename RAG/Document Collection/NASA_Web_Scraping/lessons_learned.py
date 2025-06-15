import pandas as pd
import time
from typing import Dict, List
import logging
import os
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

class NASALessonsLearned:
    def __init__(self, max_workers=4):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Number of parallel workers
        self.max_workers = max_workers
        
        # Setup Selenium with Firefox
        self.options = Options()
        self.options.add_argument('--headless')  # Run in headless mode
        
        # Update the base URL to use NASA Centers (JSC in this example)
        self.base_url = "https://llis.nasa.gov"
        self.search_url = f"{self.base_url}/search?organization=msfc&page="
        
        # Update CSV filename to reflect NASA Center data
        self.csv_filename = 'nasa_lessons_learned_centers_1.csv'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(script_dir, self.csv_filename)
        
        # Check if CSV exists, create it with headers if it doesn't
        if not os.path.exists(self.csv_path):
            self.logger.info(f"Creating new CSV file: {self.csv_filename}")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = pd.DataFrame(columns=[
                    'url', 'subject', 'abstract', 'driving_event', 
                    'lessons_learned', 'recommendations', 'evidence',
                    'program_relation', 'program_phase', 
                    'mission_directorate', 'topics'
                ]).to_csv(f, index=False)
        else:
            self.logger.info(f"Appending to existing CSV file: {self.csv_filename}")
        
        # Create a driver for URL collection
        self.driver = webdriver.Firefox(options=self.options)

    def get_lessons_urls(self, max_pages: int = 27) -> List[str]:
        lesson_urls = []
        self.logger.info(f"Starting to collect URLs from up to {max_pages} pages...")
        
        for page in range(1, max_pages + 1):
            try:
                self.logger.info(f"Collecting URLs from page {page}/{max_pages}")
                url = f"{self.search_url}{page}"
                self.driver.get(url)
                
                # Wait for the page to load
                time.sleep(3)
                
                # Check if there are any lessons on this page
                lesson_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/lesson/']")
                
                # If no lessons found, we've reached the end of available pages
                if not lesson_elements:
                    self.logger.info(f"No more lessons found on page {page}. Stopping pagination.")
                    break
                
                # Get all lesson links
                new_urls = [elem.get_attribute('href') for elem in lesson_elements]
                lesson_urls.extend(new_urls)
                
                self.logger.info(f"Found {len(new_urls)} lessons on page {page}")
                
                # Check if we're on the last page by looking at the pagination controls
                try:
                    # Look for pagination elements
                    pagination = self.driver.find_element(By.CSS_SELECTOR, ".pagination")
                    # Get the current active page number
                    active_page = pagination.find_element(By.CSS_SELECTOR, ".active").text
                    # Get all page numbers
                    page_numbers = [el.text for el in pagination.find_elements(By.CSS_SELECTOR, "li:not(.prev):not(.next) a")]
                    
                    # If the active page is the last number in the pagination, we're on the last page
                    if active_page == page_numbers[-1]:
                        self.logger.info(f"Reached the last page ({active_page}). Stopping pagination.")
                        break
                except Exception as pagination_error:
                    self.logger.warning(f"Could not determine pagination status: {pagination_error}")
                    # Continue anyway, as we'll stop if no lessons are found on the next page
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error on page {page}: {e}")
                # Don't break the loop, try the next page

        self.logger.info(f"Finished collecting URLs. Total lessons found: {len(lesson_urls)}")
        return lesson_urls

    def _get_text(self, soup: BeautifulSoup, field_name: str) -> str:
        """Helper method to extract text from a field"""
        try:
            # Find the div with ember-view class that contains the field
            for div in soup.find_all('div', class_='ember-view'):
                # Look for h3 with the field name
                h3 = div.find('h3', string=lambda x: x and field_name in x)
                if h3:
                    # Get all text content after the h3
                    content = []
                    for sibling in h3.next_siblings:
                        if sibling.name == 'h3':  # Stop if we hit another h3
                            break
                        if hasattr(sibling, 'stripped_strings'):
                            content.extend(sibling.stripped_strings)
                        elif hasattr(sibling, 'string') and sibling.string:
                            content.append(sibling.string.strip())
                    return ' '.join(filter(None, content))
            
            return "None"
        except Exception as e:
            self.logger.error(f"Error extracting {field_name}: {e}")
            return "None"

    def _get_subject(self, soup: BeautifulSoup) -> str:
        """Special method to extract subject"""
        try:
            # Find the div with ember-view class that contains 'Subject'
            for div in soup.find_all('div', class_='ember-view'):
                h3 = div.find('h3', string='Subject')
                if h3:
                    # Look for the strong tag within em
                    em = div.find('em')
                    if em:
                        strong = em.find('strong')
                        if strong:
                            return strong.get_text(strip=True)
                    # Backup: get any text content after the h3
                    content = []
                    for sibling in h3.next_siblings:
                        if hasattr(sibling, 'stripped_strings'):
                            content.extend(sibling.stripped_strings)
                    if content:
                        return ' '.join(content)
            return "None"
        except Exception as e:
            self.logger.error(f"Error extracting subject: {e}")
            return "None"

    def extract_lesson_data(self, url: str) -> Dict:
        """Extract data from a single lesson URL"""
        driver = None
        try:
            # Create a new driver for this thread
            driver = webdriver.Firefox(options=self.options)
            self.logger.debug(f"Extracting data from {url}")
            driver.get(url)
            
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ember-view"))
                )
            except Exception as wait_error:
                self.logger.warning(f"Timeout waiting for page load on {url}, attempting to extract anyway")
            
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Get subject using the special method
            subject_text = self._get_subject(soup)
            
            data = {
                'url': url,
                'subject': subject_text,
                'abstract': self._get_text(soup, 'Abstract'),
                'driving_event': self._get_text(soup, 'Driving Event'),
                'lessons_learned': self._get_text(soup, 'Lesson(s) Learned'),
                'recommendations': self._get_text(soup, 'Recommendation(s)'),
                'evidence': self._get_text(soup, 'Evidence of Recurrence Control Effectiveness'),
                'program_relation': self._get_text(soup, 'Program Relation'),
                'program_phase': self._get_text(soup, 'Program/Project Phase'),
                'mission_directorate': self._get_text(soup, 'Mission Directorate(s)'),
                'topics': self._get_text(soup, 'Topic(s)')
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return {
                'url': url,
                'subject': "Error",
                'abstract': "Error",
                'driving_event': "Error",
                'lessons_learned': "Error",
                'recommendations': "Error",
                'evidence': "Error",
                'program_relation': "Error",
                'program_phase': "Error",
                'mission_directorate': "Error",
                'topics': "Error"
            }
        finally:
            if driver:
                driver.quit()
    
    def save_to_csv(self, data: Dict):
        """Save a single lesson to CSV"""
        try:
            pd.DataFrame([data]).to_csv(
                self.csv_path, 
                mode='a', 
                header=False, 
                index=False,
                encoding='utf-8'
            )
            self.logger.debug(f"Saved data for {data['url']} to CSV")
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}")
    
    def collect_all_lessons(self) -> pd.DataFrame:
        self.logger.info("Starting collection of all lessons...")
        lesson_urls = self.get_lessons_urls()
        total_urls = len(lesson_urls)
        self.logger.info(f"Found {total_urls} lessons to process")
        
        # Check if we already have some of these URLs in the CSV
        try:
            existing_df = pd.read_csv(self.csv_path)
            existing_urls = set(existing_df['url'].tolist())
            lesson_urls = [url for url in lesson_urls if url not in existing_urls]
            self.logger.info(f"After filtering already processed URLs, {len(lesson_urls)} lessons remain to be processed")
        except Exception as e:
            self.logger.warning(f"Could not check for existing URLs: {e}")
        
        # Process lessons in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(self.extract_lesson_data, url): url for url in lesson_urls}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    data = future.result()
                    self.save_to_csv(data)
                    self.logger.info(f"Processed {i}/{len(lesson_urls)}: {url}")
                    self.logger.info(f"Subject: {data['subject'][:100]}...")
                except Exception as exc:
                    self.logger.error(f"Error processing {url}: {exc}")
        
        self.logger.info(f"Finished collecting lessons")
        return pd.read_csv(self.csv_path)
    
    def __del__(self):
        """Clean up the Selenium driver"""
        if hasattr(self, 'driver'):
            self.driver.quit()


if __name__ == "__main__":
    # You can adjust the number of parallel workers here
    scraper = NASALessonsLearned(max_workers=4)
    df = scraper.collect_all_lessons()