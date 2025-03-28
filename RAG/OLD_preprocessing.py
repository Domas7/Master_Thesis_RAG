import pandas as pd
import numpy as np
import re
from typing import List, Dict
import spacy
from datetime import datetime
import logging
from collections import defaultdict

class NASALessonsPreprocessor:
    def __init__(self, input_file: str):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load SpaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("Downloading SpaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load the data
        self.df = pd.read_csv(input_file)
        self.logger.info(f"Loaded {len(self.df)} records from {input_file}")
        
        # Initialize controlled vocabularies
        self.missions = set(['Voyager', 'Mars Observer', 'Cassini', 'Galileo', 'Mariner'])
        self.technical_domains = {
            'propulsion': ['propellant', 'thruster', 'engine', 'fuel'],
            'electronics': ['circuit', 'voltage', 'current', 'EMI'],
            'mechanical': ['structural', 'mechanism', 'actuator'],
            'thermal': ['temperature', 'thermal', 'heating'],
            'software': ['software', 'code', 'programming', 'algorithm']
        }

    def clean_text(self, text: str) -> str:
        """Clean and standardize text content"""
        if pd.isna(text) or text == 'None' or text == 'N/A':
            return np.nan
            
        # Standardize quotation marks
        text = re.sub(r'["""]', '"', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common abbreviations
        text = re.sub(r'(?<=\d)deg(?=\s?C|\s?F)', ' degrees', text)
        
        return text.strip()

    def extract_missions(self, text: str) -> List[str]:
        """Extract mentioned mission names"""
        if pd.isna(text):
            return []
        return [mission for mission in self.missions if mission in text]

    def identify_technical_domain(self, text: str) -> List[str]:
        """Identify technical domains based on keywords"""
        if pd.isna(text):
            return []
        
        domains = []
        text_lower = text.lower()
        for domain, keywords in self.technical_domains.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        return domains

    def extract_numerical_values(self, text: str) -> List[Dict]:
        """Extract numerical values with units"""
        if pd.isna(text):
            return []
        
        # Pattern for numbers with units
        pattern = r'(\d+(?:\.\d+)?)\s*(degrees?(?:\s*[CF])?|m/s|kHz|MHz|GHz|V|A)'
        matches = re.finditer(pattern, text)
        
        return [{'value': float(m.group(1)), 'unit': m.group(2)} for m in matches]

    def standardize_lists(self, text: str) -> List[str]:
        """Standardize list-type fields"""
        if pd.isna(text):
            return []
        
        # Split on common list delimiters
        items = re.split(r'[,;]|\sand\s', text)
        return [item.strip() for item in items if item.strip()]

    def process_dataframe(self) -> pd.DataFrame:
        """Apply all preprocessing steps to the dataframe"""
        self.logger.info("Starting preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = self.df.copy()
        
        # Clean text fields
        text_columns = ['subject', 'abstract', 'driving_event', 'lessons_learned', 
                       'recommendations', 'evidence']
        for col in text_columns:
            df_processed[col] = df_processed[col].apply(self.clean_text)
            self.logger.info(f"Cleaned text in {col} column")

        # Extract features
        df_processed['mentioned_missions'] = df_processed['driving_event'].apply(self.extract_missions)
        df_processed['technical_domains'] = df_processed['driving_event'].apply(self.identify_technical_domain)
        df_processed['numerical_values'] = df_processed['driving_event'].apply(self.extract_numerical_values)
        
        # Standardize list fields
        df_processed['mission_directorate'] = df_processed['mission_directorate'].apply(self.standardize_lists)
        df_processed['topics'] = df_processed['topics'].apply(self.standardize_lists)
        
        # Add metadata
        df_processed['preprocessing_timestamp'] = datetime.now()
        df_processed['lesson_id'] = df_processed['url'].apply(
            lambda x: re.search(r'/lesson/(\d+)', x).group(1) if pd.notna(x) else np.nan
        )
        
        # Create criticality flag (example logic)
        critical_keywords = ['failure', 'critical', 'severe', 'hazard', 'emergency']
        df_processed['is_critical'] = df_processed['driving_event'].apply(
            lambda x: any(keyword in str(x).lower() for keyword in critical_keywords)
        )
        
        self.logger.info("Preprocessing completed")
        return df_processed

    def save_processed_data(self, output_file: str):
        """Save the preprocessed data"""
        processed_df = self.process_dataframe()
        processed_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved preprocessed data to {output_file}")
        
        # Generate preprocessing report
        report = {
            'total_records': len(processed_df),
            'critical_lessons': processed_df['is_critical'].sum(),
            'domains_found': defaultdict(int)
        }
        
        for domains in processed_df['technical_domains']:
            for domain in domains:
                report['domains_found'][domain] += 1
                
        self.logger.info("Preprocessing Report:")
        self.logger.info(f"Total Records: {report['total_records']}")
        self.logger.info(f"Critical Lessons: {report['critical_lessons']}")
        self.logger.info("Technical Domains Distribution:")
        for domain, count in report['domains_found'].items():
            self.logger.info(f"  - {domain}: {count}")

if __name__ == "__main__":
    preprocessor = NASALessonsPreprocessor("nasa_lessons_learned_jet_propulsion_PROPER.csv")
    preprocessor.save_processed_data("nasa_lessons_learned_preprocessed.csv")
