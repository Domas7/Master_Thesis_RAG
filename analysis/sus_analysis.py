#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY', "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc5NjI1MTQsImV4cCI6MjA1MzUzODUxNH0.N0SLqiMO6KxAlf_hyNTu1W1RZ8MfltuXwtdc1o-7eAs")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_user_evaluations():
    """Fetch user evaluations from Supabase"""
    evaluations_response = supabase.table("user_evaluations").select("*").execute()
    evaluations = pd.DataFrame(evaluations_response.data)
    return evaluations

def calculate_sus_score(sus_responses):
    if not sus_responses:
        return None
    
    # Define the mapping from text responses to numeric values
    response_values = {
        "Strongly Disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly Agree": 5
    }
    
    total_score = 0
    question_scores = {}
    
    # Process each question according to SUS calculation rules
    for i in range(1, 11):
        question_key = f"sus_q{i}"
        if question_key not in sus_responses:
            continue
            
        response_text = sus_responses[question_key]
        value = response_values.get(response_text, 3)  # Default to neutral if invalid
        
        # For odd-numbered questions: (value - 1) * 2.5
        # For even-numbered questions: (5 - value) * 2.5
        if i % 2 == 1:  # Odd-numbered question
            question_score = (value - 1) * 2.5
        else:  # Even-numbered question
            question_score = (5 - value) * 2.5
            
        question_scores[question_key] = question_score
        total_score += question_score
    
    return {"total": total_score, "question_scores": question_scores}

def interpret_sus_score(score):
    """Interpret SUS score according to standard grading scale"""
    if score < 51:
        return "Poor (F)"
    elif score < 68:
        return "Below Average (D)"
    elif score < 74:
        return "Good (C)"
    elif score < 80.3:
        return "Very Good (B)"
    else:
        return "Excellent (A)"

def analyze_sus_scores(evaluations_df, output_dir='User_Evaluation_Analysis'):
    """Analyze SUS scores from user evaluations"""
    # Process SUS responses
    if evaluations_df.empty or 'sus_responses' not in evaluations_df.columns:
        print("No SUS data available for analysis")
        return None
    
    sus_results = []
    question_scores_all = []
    
    # Process each user's SUS responses
    for _, row in evaluations_df.iterrows():
        if pd.notna(row['sus_responses']):
            # Parse the JSON string if it's not already a dict
            sus_responses = row['sus_responses']
            if isinstance(sus_responses, str):
                sus_responses = json.loads(sus_responses)
            
            # Calculate SUS score
            score_data = calculate_sus_score(sus_responses)
            if score_data is not None:
                total_score = score_data["total"]
                question_scores = score_data["question_scores"]
                
                # Add interpretation
                interpretation = interpret_sus_score(total_score)
                
                sus_results.append({
                    'username': row['username'],
                    'sus_score': total_score,
                    'interpretation': interpretation,
                    'timestamp': row['timestamp']
                })
                
                # Add question scores for detailed analysis
                q_scores = {'username': row['username']}
                q_scores.update(question_scores)
                question_scores_all.append(q_scores)
    
    if not sus_results:
        print("No valid SUS data available")
        return None
    
    # Convert to DataFrame
    sus_df = pd.DataFrame(sus_results)
    question_df = pd.DataFrame(question_scores_all)
    
    # Summary statistics
    summary = {
        'mean_score': sus_df['sus_score'].mean(),
        'median_score': sus_df['sus_score'].median(),
        'min_score': sus_df['sus_score'].min(),
        'max_score': sus_df['sus_score'].max(),
        'std_dev': sus_df['sus_score'].std(),
        'num_respondents': len(sus_df),
        'average_interpretation': interpret_sus_score(sus_df['sus_score'].mean())
    }
    
    print("\n=== SUS SCORE SUMMARY ===")
    print(f"Average SUS Score: {summary['mean_score']:.2f} - {summary['average_interpretation']}")
    print(f"Median SUS Score: {summary['median_score']:.2f}")
    print(f"Range: {summary['min_score']:.2f} to {summary['max_score']:.2f}")
    print(f"Standard Deviation: {summary['std_dev']:.2f}")
    print(f"Number of Respondents: {summary['num_respondents']}")
    
    return summary

def main():
    """Main function to run SUS analysis"""
    print("Fetching user evaluations from Supabase...")
    evaluations_df = fetch_user_evaluations()
    
    print(f"Retrieved {len(evaluations_df)} evaluation records")
    
    # Analyze SUS scores
    print("Analyzing SUS scores...")
    summary = analyze_sus_scores(evaluations_df)
    
    if summary:
        print("\nSUS analysis complete!")
        print(f"Results saved to 'User_Evaluation_Analysis' directory")
    else:
        print("No data available for SUS analysis")

if __name__ == "__main__":
    main() 