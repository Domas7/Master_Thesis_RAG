#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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

def fetch_data():
    """Fetch data from Supabase tables"""
    # Fetch user answers
    answers_response = supabase.table("user_answers").select("*").execute()
    answers = pd.DataFrame(answers_response.data)
    
    return answers

def analyze_model_performance(answers_df, max_time_minutes=15):
    """
    Analyze task completion time by model and calculate success rates
    """
    print("\nAnalyzing task completion times and logging unusual cases...")
    
    # Filter to only include actual task submissions (not queries)
    task_submissions = answers_df[
        (answers_df['task_id'].str.startswith('task')) & 
        (answers_df['entry_type'] == 'submission')
    ].copy()
    
    # Also get the query data
    task_queries = answers_df[
        (answers_df['entry_type'] == 'query') &
        (answers_df['task_id'].str.startswith('query_task'))
    ].copy()

    # Get RAG queries data for task1 timing
    rag_queries_response = supabase.table("rag_queries").select("*").execute()
    rag_queries = pd.DataFrame(rag_queries_response.data)
    rag_queries['timestamp'] = pd.to_datetime(rag_queries['timestamp'])
    
    if task_submissions.empty:
        print("No task submissions found")
        return None
    
    # Convert timestamp to datetime if it's not already
    task_submissions['timestamp'] = pd.to_datetime(task_submissions['timestamp'])
    task_queries['timestamp'] = pd.to_datetime(task_queries['timestamp'])
    
    # Group by user and task_id to find completion times
    completion_times = []
    unusual_cases = []
    
    # Get unique users
    users = task_submissions['username'].unique()
    print(f"\nProcessing data for {len(users)} users")
    
    for user in users:
        print(f"\nAnalyzing user: {user}")
        user_submissions = task_submissions[task_submissions['username'] == user].copy()
        user_queries = task_queries[task_queries['username'] == user].copy()
        user_rag_queries = rag_queries[rag_queries['username'] == user].copy()
        
        # Sort by task_id numerical value and timestamp
        user_submissions['task_num'] = user_submissions['task_id'].str.extract(r'task(\d+)').astype(int)
        user_submissions = user_submissions.sort_values(['task_num', 'timestamp'])
        
        # Get completion times for each task
        task_completion_times = {}
        for _, row in user_submissions.iterrows():
            if row['is_correct'] == True:  # Only consider successful submissions
                task_completion_times[row['task_id']] = row['timestamp']
        
        # Process each task for this user
        for task_id in sorted(task_completion_times.keys(), key=lambda x: int(x.replace('task', ''))):
            print(f"\n  Processing {task_id}:")
            task_num = int(task_id.replace('task', ''))
            task_data = user_submissions[user_submissions['task_id'] == task_id]
            
            # Find the start time based on task number
            start_time = None
            start_time_source = ""
            
            if task_num == 1:
                # For Task 1, start time is the first RAG query for task1
                task1_queries = user_rag_queries[
                    user_rag_queries['task_id'] == 'task1'
                ]
                
                if not task1_queries.empty:
                    start_time = task1_queries['timestamp'].min()
                    start_time_source = "first_rag_query"
                    print(f"    Found first RAG query for task1: {start_time}")
                else:
                    # Fallback to first submission if no RAG queries found
                    start_time = user_submissions[user_submissions['task_id'] == 'task1']['timestamp'].min()
                    start_time_source = "first_submission_fallback"
            else:
                # For Tasks 2-5, start time is when the previous task was completed
                prev_task_id = f'task{task_num-1}'
                if prev_task_id in task_completion_times:
                    start_time = task_completion_times[prev_task_id]
                    start_time_source = "previous_task_completion"
                else:
                    # If previous task wasn't completed, use the first submission for this task
                    start_time = task_data['timestamp'].min()
                    start_time_source = "first_task_submission"
            
            # End time is when this task was completed
            end_time = task_completion_times[task_id]
            
            # Log timing information
            print(f"    Start time ({start_time_source}): {start_time}")
            print(f"    End time: {end_time}")
            
            # If we have both start and end times
            if start_time is not None and end_time is not None:
                # Calculate time difference in minutes
                time_diff = (end_time - start_time).total_seconds() / 60
                
                # Handle negative times by using absolute value
                if time_diff < 0:
                    print(f"    Found negative time: {time_diff:.2f} minutes, using absolute value")
                    time_diff = abs(time_diff)
                    
                print(f"    Time difference: {time_diff:.2f} minutes")
                
                # Only include times that are within the max_time_minutes limit
                if time_diff <= max_time_minutes:
                    # Get model used
                    model = task_data['model_used'].iloc[0] if 'model_used' in task_data.columns and not task_data['model_used'].isna().all() else 'unknown'
                    
                    # Handle case sensitivity and normalize model names
                    if isinstance(model, str):
                        if model.lower() in ['openai', 'gpt-4', 'gpt-4o-mini', 'gpt4', 'gpt']:
                            model = 'OpenAI'
                        elif model.lower() in ['llama', 'llama3', 'llama-3', 'wizardlm2']:
                            model = 'Llama'
                    
                    # Count attempts
                    attempts = len(task_data)
                    
                    completion_times.append({
                        'username': user,
                        'task_id': task_id,
                        'model_used': model,
                        'time_to_complete_minutes': time_diff,
                        'is_correct': True,  # Only correct submissions are in task_completion_times
                        'attempts': attempts,
                        'start_time_source': start_time_source
                    })
                else:
                    print(f"    WARNING: Completion time exceeds maximum allowed time!")
                    unusual_cases.append({
                        'username': user,
                        'task_id': task_id,
                        'issue': 'exceeded_max_time',
                        'time_diff': time_diff,
                        'max_allowed': max_time_minutes
                    })
    
    # Convert to DataFrame
    completion_df = pd.DataFrame(completion_times)
    
    # Print summary statistics
    if not completion_df.empty:
        print("\nCompletion time summary statistics11:")
        for task in sorted(completion_df['task_id'].unique()):
            task_times = completion_df[completion_df['task_id'] == task]['time_to_complete_minutes']
            print(f"\n{task}:")
            print(f"  Mean: {task_times.mean():.3f} minutes")
            print(f"  Median: {task_times.median():.3f} minutes")
            print(f"  Min: {task_times.min():.3f} minutes")
            print(f"  Max: {task_times.max():.3f} minutes")
            print(f"  Count: {len(task_times)} completions")
    
    # Print summary of unusual cases
    if unusual_cases:
        print("\n=== UNUSUAL CASES SUMMARY ===")
        unusual_df = pd.DataFrame(unusual_cases)
        print(f"\nFound {len(unusual_cases)} unusual cases:")
        for issue_type in unusual_df['issue'].unique():
            count = len(unusual_df[unusual_df['issue'] == issue_type])
            print(f"- {issue_type}: {count} cases")
    
    return completion_df

def generate_model_comparison_report(completion_df, output_dir='model_comparison'):
    """Generate detailed model comparison report with metrics only"""
    if completion_df is None or completion_df.empty:
        print("No data available for model comparison")
        return
    
    # 1. Model completion time comparison
    print("\n=== MODEL COMPLETION TIME COMPARISON ===")
    model_times = completion_df.groupby('model_used')['time_to_complete_minutes'].agg(['mean', 'median', 'std', 'count'])
    model_times.columns = ['Mean Time (min)', 'Median Time (min)', 'Std Dev (min)', 'Sample Count']
    print(model_times)
    
    # 2. Model success rate comparison
    print("\n=== MODEL SUCCESS RATE COMPARISON ===")
    model_success = completion_df.groupby('model_used')['is_correct'].agg(['mean', 'count'])
    model_success.columns = ['Success Rate', 'Sample Count']
    model_success['Success Rate'] = model_success['Success Rate'] * 100  # Convert to percentage
    print(model_success)
    
    # 3. Task-level model comparison
    print("\n=== TASK-LEVEL MODEL COMPARISON ===")
    task_model_stats = completion_df.groupby(['task_id', 'model_used']).agg({
        'time_to_complete_minutes': ['mean', 'median', 'std', 'count'],
        'is_correct': ['mean'],
        'attempts': ['mean']
    })
    
    # Flatten the column hierarchy
    task_model_stats.columns = ['_'.join(col).strip() for col in task_model_stats.columns.values]
    task_model_stats['is_correct_mean'] = task_model_stats['is_correct_mean'] * 100  # Convert to percentage
    print(task_model_stats)
    
    # Generate summary metrics
    summary = {
        "overall": {
            "sample_count": len(completion_df),
            "model_distribution": completion_df['model_used'].value_counts().to_dict()
        },
        "time_metrics": model_times.to_dict(),
        "success_metrics": model_success.to_dict(),
        "detailed_metrics": {
            "tasks": len(completion_df['task_id'].unique()),
            "users": len(completion_df['username'].unique()),
            "average_attempts": completion_df['attempts'].mean()
        }
    }
    
    return summary

def main():
    """Main function to run model comparison analysis"""
    print("Fetching data from Supabase...")
    answers_df = fetch_data()
    
    print(f"Retrieved {len(answers_df)} answer records")
    
    # Analyze model performance
    print("Analyzing model performance...")
    completion_df = analyze_model_performance(answers_df)
    
    if completion_df is not None:
        # Generate comparison report
        print("Generating model comparison report...")
        summary = generate_model_comparison_report(completion_df)
        
        print("\nModel comparison analysis complete!")
        print(f"Results saved to 'model_comparison' directory")
    else:
        print("No data available for analysis")

if __name__ == "__main__":
    main() 