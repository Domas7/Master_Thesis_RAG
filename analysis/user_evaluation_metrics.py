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
    # Fetch user evaluations
    evaluations_response = supabase.table("user_evaluations").select("*").execute()
    evaluations = pd.DataFrame(evaluations_response.data)
    
    # Fetch user answers
    answers_response = supabase.table("user_answers").select("*").execute()
    answers = pd.DataFrame(answers_response.data)
    
    return evaluations, answers

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
            
        total_score += question_score
    
    return total_score

def analyze_task_completion_times(answers_df, max_time_minutes=10):
    # Get RAG queries from the database
    rag_queries_response = supabase.table("rag_queries").select("*").execute()
    rag_queries = pd.DataFrame(rag_queries_response.data)
    
    # Filter to only include actual task submissions (not queries)
    task_submissions = answers_df[
        (answers_df['task_id'].str.startswith('task')) & 
        (answers_df['entry_type'] == 'submission')
    ].copy()
    
    if task_submissions.empty:
        print("No task submissions found")
        return None
    
    # Convert timestamp to datetime if it's not already
    task_submissions['timestamp'] = pd.to_datetime(task_submissions['timestamp'])
    rag_queries['timestamp'] = pd.to_datetime(rag_queries['timestamp'])
    
    # Group by user and task_id to find completion times
    completion_times = []
    
    # Get unique users
    users = task_submissions['username'].unique()
    print(f"\nAnalyzing completion times for {len(users)} users")
    
    for user in users:
        print(f"\nProcessing user: {user}")
        user_submissions = task_submissions[task_submissions['username'] == user].copy()
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
            task_num = int(task_id.replace('task', ''))
            task_data = user_submissions[user_submissions['task_id'] == task_id]
            
            # Find the start time based on task number
            start_time = None
            
            if task_num == 1:
                # For Task 1, start time is the first ever query from rag_queries
                if not user_rag_queries.empty:
                    start_time = user_rag_queries['timestamp'].min()
                    print(f"  Task 1 - First ever query time: {start_time}")
                else:
                    print(f"  Task 1 - No queries found, using first submission time")
                    start_time = user_submissions[user_submissions['task_id'] == 'task1']['timestamp'].min()
            else:
                # For Tasks 2-5, start time is when the previous task was completed
                prev_task_id = f'task{task_num-1}'
                if prev_task_id in task_completion_times:
                    start_time = task_completion_times[prev_task_id]
                else:
                    start_time = task_data['timestamp'].min()
            
            # End time is when this task was completed
            end_time = task_completion_times[task_id]
            print(f"  {task_id} - Start time: {start_time}, End time: {end_time}")
            
            # If we have both start and end times
            if start_time is not None and end_time is not None:
                # Calculate time difference in minutes with higher precision
                time_diff = (end_time - start_time).total_seconds() / 60.0
                
                # Handle negative times by using absolute value
                if time_diff < 0:
                    print(f"  {task_id} - Found negative time: {time_diff:.3f} minutes, using absolute value")
                    time_diff = abs(time_diff)
                
                print(f"  {task_id} - Time difference: {time_diff:.3f} minutes")
                
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
                    
                    completion_times.append({
                        'username': user,
                        'task_id': task_id,
                        'model_used': model,
                        'time_to_complete_minutes': time_diff,
                        'is_correct': True  # Only correct submissions are in task_completion_times
                    })
    
    # Convert to DataFrame
    completion_df = pd.DataFrame(completion_times)
    
    # Print summary statistics
    if not completion_df.empty:
        print("\nCompletion time summary statistics:")
        for task in sorted(completion_df['task_id'].unique()):
            task_times = completion_df[completion_df['task_id'] == task]['time_to_complete_minutes']
            print(f"\n{task}:")
            print(f"  Mean: {task_times.mean():.3f} minutes")
            print(f"  Median: {task_times.median():.3f} minutes")
            print(f"  Min: {task_times.min():.3f} minutes")
            print(f"  Max: {task_times.max():.3f} minutes")
            print(f"  Count: {len(task_times)} completions")
    
    return completion_df

def generate_metrics(evaluations_df, answers_df):
    """Generate metrics from the evaluation and answers data"""
    metrics = {}
    
    # 1. Calculate SUS scores for each user
    if not evaluations_df.empty and 'sus_responses' in evaluations_df.columns:
        sus_scores = []
        for _, row in evaluations_df.iterrows():
            if pd.notna(row['sus_responses']):
                # Parse the JSON string if it's not already a dict
                sus_responses = row['sus_responses']
                if isinstance(sus_responses, str):
                    sus_responses = json.loads(sus_responses)
                
                score = calculate_sus_score(sus_responses)
                if score is not None:
                    sus_scores.append({
                        'username': row['username'],
                        'sus_score': score
                    })
        
        # Create DataFrame and calculate stats
        if sus_scores:
            sus_df = pd.DataFrame(sus_scores)
            metrics['sus'] = {
                'mean_score': sus_df['sus_score'].mean(),
                'median_score': sus_df['sus_score'].median(),
                'min_score': sus_df['sus_score'].min(),
                'max_score': sus_df['sus_score'].max(),
                'scores_by_user': sus_df.to_dict(orient='records')
            }
            
            # SUS score interpretation
            avg_score = metrics['sus']['mean_score']
            if avg_score < 51:
                metrics['sus']['interpretation'] = "Poor (F)"
            elif avg_score < 68:
                metrics['sus']['interpretation'] = "Below Average (D)"
            elif avg_score < 74:
                metrics['sus']['interpretation'] = "Good (C)"
            elif avg_score < 80.3:
                metrics['sus']['interpretation'] = "Very Good (B)"
            else:
                metrics['sus']['interpretation'] = "Excellent (A)"
    
    # 2. Analyze task completion times
    completion_df = analyze_task_completion_times(answers_df)
    if completion_df is not None and not completion_df.empty:
        # Store completion_df in metrics for visualization
        metrics['completion_data'] = completion_df.to_dict('records')
        
        # Overall stats
        metrics['task_completion'] = {
            'mean_time_minutes': completion_df['time_to_complete_minutes'].mean(),
            'median_time_minutes': completion_df['time_to_complete_minutes'].median(),
            'success_rate': completion_df['is_correct'].mean() * 100  # as percentage
        }
        
        # By task
        task_stats = completion_df.groupby('task_id').agg({
            'time_to_complete_minutes': ['mean', 'median', 'count'],
            'is_correct': ['mean']
        })
        
        task_stats.columns = ['mean_time', 'median_time', 'attempts', 'success_rate']
        task_stats['success_rate'] = task_stats['success_rate'] * 100  # as percentage
        
        metrics['task_stats'] = task_stats.reset_index().to_dict(orient='records')
        
        # By model
        if 'model_used' in completion_df.columns:
            model_stats = completion_df.groupby(['task_id', 'model_used']).agg({
                'time_to_complete_minutes': ['mean', 'min', 'max', 'count']
            }).reset_index()
            
            # Flatten the column names
            model_stats.columns = ['task_id', 'model_used', 'mean', 'min', 'max', 'count']
            
            metrics['model_comparison'] = model_stats.reset_index().to_dict(orient='records')
    
    # 3. Task difficulty analysis
    if not evaluations_df.empty and 'task_difficulty' in evaluations_df.columns:
        difficulty_responses = []
        
        for _, row in evaluations_df.iterrows():
            if pd.notna(row['task_difficulty']):
                # Parse the JSON string if it's not already a dict
                task_difficulty = row['task_difficulty']
                if isinstance(task_difficulty, str):
                    task_difficulty = json.loads(task_difficulty)
                
                # Ensure we have entries for all tasks
                for task_id in [f'task{i}' for i in range(1, 6)]:
                    difficulty = task_difficulty.get(task_id, None)
                    if difficulty is not None:  # Only add if we have a response
                        difficulty_responses.append({
                            'username': row['username'],
                            'task_id': task_id,
                            'difficulty': difficulty
                        })
        
        if difficulty_responses:
            difficulty_df = pd.DataFrame(difficulty_responses)
            
            # Count occurrences of each difficulty level by task
            difficulty_counts = pd.crosstab(difficulty_df['task_id'], difficulty_df['difficulty'])
            
            # Calculate percentages
            difficulty_percentages = difficulty_counts.div(difficulty_counts.sum(axis=1), axis=0) * 100
            
            # Combine counts and percentages
            result_data = []
            for task_id in sorted([f'task{i}' for i in range(1, 6)]):  # Ensure all tasks are included
                row_data = {
                    'task_id': task_id
                }
                # Add raw counts
                if task_id in difficulty_counts.index:
                    for col in difficulty_counts.columns:
                        row_data[col] = difficulty_counts.loc[task_id, col]
                    # Add percentages
                    for col in difficulty_percentages.columns:
                        row_data[f"{col}_pct"] = difficulty_percentages.loc[task_id, col]
                else:
                    # Add zero counts and percentages for missing tasks
                    for col in ['Too Easy', 'Easy', 'Just Right', 'Challenging', 'Too Difficult']:
                        row_data[col] = 0
                        row_data[f"{col}_pct"] = 0
                
                result_data.append(row_data)
            
            metrics['task_difficulty'] = result_data
    
    # 4. AI Assistant performance metrics
    if not evaluations_df.empty:
        ai_metrics = {}
        
        # Process AI helpfulness ratings
        if 'ai_helpfulness' in evaluations_df.columns:
            helpfulness_counts = evaluations_df['ai_helpfulness'].value_counts().to_dict()
            ai_metrics['helpfulness'] = helpfulness_counts
        
        # Process AI relevance ratings
        if 'ai_relevance' in evaluations_df.columns:
            relevance_counts = evaluations_df['ai_relevance'].value_counts().to_dict()
            ai_metrics['relevance'] = relevance_counts
        
        metrics['ai_performance'] = ai_metrics
    
    # 5. Skipped tasks analysis
    if not evaluations_df.empty and 'skipped_tasks' in evaluations_df.columns:
        skipped_tasks = []
        
        for _, row in evaluations_df.iterrows():
            if pd.notna(row['skipped_tasks']):
                # Parse the JSON string if it's not already a list
                tasks = row['skipped_tasks']
                if isinstance(tasks, str):
                    tasks = json.loads(tasks)
                
                if tasks:  # Check if the list is not empty
                    for task in tasks:
                        skipped_tasks.append({
                            'username': row['username'],
                            'task_id': task
                        })
        
        if skipped_tasks:
            skipped_df = pd.DataFrame(skipped_tasks)
            skip_counts = skipped_df['task_id'].value_counts().to_dict()
            metrics['skipped_tasks'] = skip_counts
    
    return metrics

def visualize_metrics(metrics, output_dir='User_Evaluation_Analysis'):
    """Create visualizations from the metrics"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Get RAG queries data for processing time analysis
    rag_queries_response = supabase.table("rag_queries").select("*").execute()
    rag_queries = pd.DataFrame(rag_queries_response.data)
    
    # Get answers data for attempt analysis
    answers_response = supabase.table("user_answers").select("*").execute()
    answers_df = pd.DataFrame(answers_response.data)
    
    # Normalize model names in rag_queries
    rag_queries['model'] = rag_queries['model'].apply(
        lambda x: 'OpenAI' if str(x).lower() in ['openai', 'gpt-4', 'gpt4', 'gpt'] 
        else 'Llama' if str(x).lower() in ['llama', 'llama3', 'llama-3'] 
        else x
    )
    
    # Extract task number from task_id in queries (if available)
    rag_queries['task_num'] = rag_queries['task_id'].str.extract(r'task(\d+)').fillna('0')
    # Filter out non-task queries and ensure we only have tasks 1-5
    rag_queries = rag_queries[rag_queries['task_num'].astype(float).between(1, 5)]
    rag_queries['task_id'] = 'task' + rag_queries['task_num'].astype(str)
    
    # 1. Task Difficulty Visualization
    if 'task_difficulty' in metrics:
        difficulty_df = pd.DataFrame(metrics['task_difficulty'])
        difficulty_levels = ['Too Easy_pct', 'Easy_pct', 'Just Right_pct', 'Challenging_pct', 'Too Difficult_pct']
        available_levels = [col for col in difficulty_levels if col in difficulty_df.columns]
        
        if available_levels:
            plt.figure(figsize=(14, 8))
            
            # Sort tasks to ensure they're in order
            difficulty_df['task_num'] = difficulty_df['task_id'].str.extract('(\d+)').astype(int)
            difficulty_df = difficulty_df.sort_values('task_num')
            difficulty_df = difficulty_df.drop('task_num', axis=1)
            
            # Prepare data for plotting
            difficulty_melted = pd.melt(difficulty_df, 
                                      id_vars=['task_id'], 
                                      value_vars=available_levels,
                                      var_name='difficulty_level', 
                                      value_name='percentage')
            
            difficulty_melted['difficulty_level'] = difficulty_melted['difficulty_level'].str.replace('_pct', '')
            
            # Create stacked bar chart
            ax = plt.gca()
            difficulty_pivot = difficulty_melted.pivot(index='task_id', columns='difficulty_level', values='percentage')
            
            # Reorder the columns in desired order
            desired_order = ['Too Easy', 'Easy', 'Just Right', 'Challenging', 'Too Difficult']
            #desired_order = ['Too Difficult', 'Challenging', 'Just Right', 'Easy', 'Too Easy']
            difficulty_pivot = difficulty_pivot[desired_order]
            
            difficulty_pivot.plot(kind='bar', stacked=True, ax=ax)
            
            # Increase font sizes
            plt.rcParams.update({'font.size': 16})  # Increase base font size
            
            # Set y-axis to go from 0 to 100
            plt.ylim(0, 100)
            
            # Add percentage labels on the bars with larger font
            prev_heights = np.zeros(len(difficulty_pivot))
            for i, col in enumerate(difficulty_pivot.columns):
                heights = difficulty_pivot[col].values
                for j, h in enumerate(heights):
                    if h > 0:  # Only show label if percentage is greater than 0
                        plt.text(j, prev_heights[j] + h/2, f'{h:.0f}%', 
                               ha='center', va='center', fontsize=14)
                prev_heights += heights
            
            plt.title('Task Difficulty Distribution', fontsize=20, pad=30)
            plt.ylabel('Percentage of Responses', fontsize=16)
            plt.xlabel('Task', fontsize=16)
            plt.legend(title='Difficulty Level', bbox_to_anchor=(0.5, 1.25), loc='center', ncol=5,
                      fontsize=14, title_fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/task_difficulty.svg', bbox_inches='tight', dpi=300)
            plt.close()
    
    # 2. Task Completion Times Visualization
    if 'completion_data' in metrics:
        completion_df = pd.DataFrame(metrics['completion_data'])
        
        # Original task completion times visualization
        plt.figure(figsize=(14, 8))
        
        # Increase font sizes
        plt.rcParams.update({'font.size': 16})
        
        # Filter out times exceeding 50 minutes
        completion_df_filtered = completion_df[completion_df['time_to_complete_minutes'] <= 50]
        
        # Sort tasks in correct order
        completion_df_filtered['task_num'] = completion_df_filtered['task_id'].str.extract('(\d+)').astype(int)
        completion_df_filtered = completion_df_filtered.sort_values('task_num')
        
        # Calculate statistics for each task
        task_stats = completion_df_filtered.groupby('task_id').agg({
            'time_to_complete_minutes': ['mean', 'min', 'max', 'count']
        }).reset_index()
        task_stats.columns = ['task_id', 'mean', 'min', 'max', 'count']
        
        # Sort task_stats in correct order
        task_stats['task_num'] = task_stats['task_id'].str.extract('(\d+)').astype(int)
        task_stats = task_stats.sort_values('task_num')
        
        # Create box plot with larger fonts
        sns.boxplot(data=completion_df_filtered, x='task_id', y='time_to_complete_minutes', order=sorted(completion_df_filtered['task_id'].unique()))
        
        # Add individual points for each user
        sns.stripplot(data=completion_df_filtered, x='task_id', y='time_to_complete_minutes', 
                     color='red', alpha=0.3, jitter=0.2, size=8,
                     order=sorted(completion_df_filtered['task_id'].unique()))
        
        # Add mean completion time as text above each box
        for i, task in enumerate(completion_df_filtered['task_id'].unique()):
            task_data = completion_df_filtered[completion_df_filtered['task_id'] == task]
            mean_time = task_data['time_to_complete_minutes'].mean()
            max_time = task_data['time_to_complete_minutes'].max()
            plt.text(i, max_time + 0.5, f'Mean: {mean_time:.1f}m', 
                    horizontalalignment='center', color='darkblue')
        
        plt.title('Task Completion Times (Times exceeding 10 minutes filtered out)')
        plt.ylabel('Time to Complete (minutes)')
        plt.xlabel('Task')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0, top=11)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/task_completion_times.svg')
        plt.close()

        # 3. Model Comparison Visualization
        # Normalize model names
        completion_df['model_used'] = completion_df['model_used'].apply(
            lambda x: 'OpenAI' if str(x).lower() in ['openai', 'gpt-4', 'gpt4', 'gpt'] 
            else 'Llama' if str(x).lower() in ['llama', 'llama3', 'llama-3'] 
            else x
        )
        
        # Calculate mean times for each task and model
        model_means = completion_df.groupby(['task_id', 'model_used']).agg({
            'time_to_complete_minutes': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten the column names
        model_means.columns = ['task_id', 'model_used', 'mean', 'min', 'max', 'count']
        
        # Create the comparison plot
        plt.figure(figsize=(14, 8))
        
        # Set larger font size globally
        plt.rcParams.update({'font.size': 14})
        
        # Create grouped bar plot
        bar_width = 0.35
        tasks = sorted(completion_df['task_id'].unique())
        x = np.arange(len(tasks))
        
        # Plot bars for each model
        for i, model in enumerate(['OpenAI', 'Llama']):
            model_data = model_means[model_means['model_used'] == model]
            
            # Calculate error bar heights (distance from mean to min/max)
            yerr = np.array([
                model_data['mean'] - model_data['min'],  # Distance from mean to min
                model_data['max'] - model_data['mean']   # Distance from mean to max
            ])
            
            bars = plt.bar(x + (i - 0.5) * bar_width, model_data['mean'], 
                         bar_width, label=model,
                         yerr=yerr, capsize=5)
            
            # Add value labels on top of bars with larger font
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                count = model_data.iloc[idx]['count']
                # Adjust x position based on model
                x_offset = -0.12 if model == 'OpenAI' else 0.12  # Move OpenAI labels left, Llama labels right
                plt.text(bar.get_x() + bar_width/2 + x_offset, height,
                        f'{height:.1f}m\n(n={int(count)})',
                        ha='center', va='bottom',
                        fontsize=12)
        
        plt.title('Average Task Completion Times by Model\n(Attempts exceeding 14 minutes filtered out)', 
                 pad=20, fontsize=16)
        plt.xlabel('Task', fontsize=14)
        plt.ylabel('Average Time to Complete (minutes)', fontsize=14)
        plt.xticks(x, tasks, fontsize=12)
        plt.legend(fontsize=12, loc='upper right')  # Move legend to upper right to avoid overlap
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Ensure y-axis starts at 0 and extends to cover all data points
        plt.ylim(bottom=0, top=max(model_means['max'].max() * 1.1, 14))  # Ensure y-axis goes up to at least 14
        
        # Make y-axis labels larger
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_completion_comparison.svg')
        plt.close()

    # 4. Model Processing Time Comparison
    if not rag_queries.empty:
        # Calculate mean processing times for each task and model
        processing_means = rag_queries.groupby(['task_id', 'model'])['processing_time'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Create the comparison plot
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar plot
        bar_width = 0.35
        tasks = sorted(rag_queries['task_id'].unique())  # This will now only include task1-task5
        x = np.arange(len(tasks))
        
        # Plot bars for each model
        for i, model in enumerate(['OpenAI', 'Llama']):
            model_data = processing_means[processing_means['model'] == model]
            # Ensure data is sorted and aligned with x-axis
            model_data = model_data.set_index('task_id').reindex(tasks).reset_index()
            # Replace NaN with 0 for plotting
            model_data = model_data.fillna(0)
            
            bars = plt.bar(x + (i - 0.5) * bar_width, model_data['mean'], 
                         bar_width, label=model,
                         yerr=model_data['std'], capsize=5)
            
            # Add value labels on top of bars
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                count = model_data.iloc[idx]['count']
                if height > 0:  # Only show label if there's data
                    plt.text(bar.get_x() + bar_width/2, height,
                            f'{height:.1f}s\n(n={int(count)})',
                            ha='center', va='bottom')
        
        plt.title('Mean Processing Time by Model and Task')
        plt.xlabel('Task')
        plt.ylabel('Mean Processing Time (seconds)')
        plt.xticks(x, tasks)
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Ensure y-axis starts at 0
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_processing_comparison.svg')
        plt.close()

    # 5. Task Attempts Analysis
    if not answers_df.empty:
        # Filter for submission entries only
        submissions = answers_df[
            (answers_df['entry_type'] == 'submission') & 
            (answers_df['task_id'].str.startswith('task'))
        ].copy()
        
        # Normalize model names in submissions
        submissions['model_used'] = submissions['model_used'].apply(
            lambda x: 'OpenAI' if str(x).lower() in ['openai', 'gpt-4', 'gpt4', 'gpt'] 
            else 'Llama' if str(x).lower() in ['llama', 'llama3', 'llama-3'] 
            else x
        )
        
        # Get skipped tasks
        skipped = answers_df[
            (answers_df['entry_type'] == 'query') & 
            (answers_df['query'].str.contains('Task skipped', na=False))
        ].copy()
        
        # Extract task number from skipped tasks
        skipped['task_id'] = skipped['query'].str.extract(r'Task skipped - (task\d+)')
        
        # Calculate attempts per task and model
        attempts_data = []
        tasks = [f'task{i}' for i in range(1, 6)]
        models = ['OpenAI', 'Llama']
        
        for task in tasks:
            for model in models:
                # Get submissions for this task and model
                task_submissions = submissions[
                    (submissions['task_id'] == task) & 
                    (submissions['model_used'] == model)
                ]
                
                if not task_submissions.empty:
                    # Count number of submissions per user until success
                    user_attempts = []
                    for user in task_submissions['username'].unique():
                        user_tries = task_submissions[task_submissions['username'] == user]
                        # Find index of first success (if any)
                        if True in user_tries['is_correct'].values:
                            success_idx = user_tries['is_correct'].values.argmax()
                            attempts = success_idx + 1  # Add 1 because even first try counts as an attempt
                        else:
                            # If no success, count all attempts
                            attempts = len(user_tries)
                        user_attempts.append(attempts)
                    
                    # Calculate statistics
                    attempts_data.append({
                        'task_id': task,
                        'model_used': model,
                        'mean_attempts': np.mean(user_attempts),
                        'min_attempts': np.min(user_attempts),
                        'max_attempts': np.max(user_attempts),
                        'users': len(user_attempts),
                        'skipped': len(skipped[
                            (skipped['task_id'] == task) & 
                            (skipped['model_used'] == model)
                        ])
                    })
        
        # Create DataFrame for plotting
        attempts_df = pd.DataFrame(attempts_data)
        
        # Create the visualization with increased spacing and fonts
        plt.figure(figsize=(14, 8))
        plt.rcParams.update({'font.size': 14})
        
        # Create grouped bar plot with wider bars
        bar_width = 0.4  # Increased from 0.35
        x = np.arange(len(tasks)) * 1.2  # Increased spacing between task groups
        
        # Plot bars for each model
        for i, model in enumerate(['OpenAI', 'Llama']):
            model_data = attempts_df[attempts_df['model_used'] == model]
            # Ensure data is sorted and aligned with x-axis
            model_data = model_data.set_index('task_id').reindex(tasks).reset_index()
            model_data = model_data.fillna(0)
            
            # Calculate error bar heights
            yerr = np.array([
                model_data['mean_attempts'] - model_data['min_attempts'],
                model_data['max_attempts'] - model_data['mean_attempts']
            ])
            
            bars = plt.bar(x + (i - 0.5) * bar_width, model_data['mean_attempts'], 
                         bar_width, label=model,
                         yerr=yerr, capsize=5)
            
            # Add value labels with larger font
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                users = model_data.iloc[idx]['users']
                skipped = model_data.iloc[idx]['skipped']
                max_attempts = model_data.iloc[idx]['max_attempts']
                if height > 0:
                    label_text = f'avg={height:.1f}\nmax={int(max_attempts)}\n(n={int(users)}'
                    if skipped > 0:
                        label_text += f'\nskipped={int(skipped)}'
                    label_text += ')'
                    plt.text(bar.get_x() + bar_width/2, max_attempts,
                            label_text,
                            ha='center', va='bottom',
                            fontsize=12)
        
        plt.title('Analysis of Task Completion Attempts by Model\nShowing Average, Max Attempts, and Skipped Tasks', 
                 pad=30, fontsize=16)
        plt.xlabel('Task', fontsize=14)
        plt.ylabel('Number of Attempts', fontsize=14)
        plt.xticks(x, tasks, fontsize=12)
        plt.legend(fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        plt.ylim(bottom=1, top=8)
        
        # Make y-axis labels larger
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/task_attempts_analysis.svg')
        plt.close()

def visualize_feedback_responses(evaluations_df, output_dir):
    """Create a 2x2 grid of bar plots showing feedback response distributions"""
    if evaluations_df.empty:
        return

    # Define base colors for each subplot (matching task difficulty colors)
    base_colors = {
        'ai_helpfulness': '#8B4513',    # Orange-brown
        'ai_relevance': '#1f77b4',      # Blue
        'retrieval_quality': '#9467bd',  # Purple
        'traditional_comparison': '#d62728'  # Red
    }

    # Define the questions and their possible responses
    feedback_questions = {
        'ai_helpfulness': {
            'title': 'How helpful was the AI assistant?',
            'options': ["Not helpful", "Slightly helpful", "Moderately helpful", "Very helpful", "Extremely helpful"]
        },
        'ai_relevance': {
            'title': 'How relevant were the AI\'s responses?',
            'options': ["Not relevant", "Somewhat relevant", "Moderately relevant", "Very relevant", "Extremely relevant"]
        },
        'retrieval_quality': {
            'title': 'Quality of Retrieved Information',
            'options': ["Poor", "Fair", "Good", "Very Good", "Excellent"]
        },
        'traditional_comparison': {
            'title': 'Compared to Traditional Search Methods',
            'options': ["Much worse", "Worse", "About the same", "Better", "Much better"]
        }
    }

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('User Feedback Analysis', fontsize=16, y=0.95)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each question
    for idx, (column, question_info) in enumerate(feedback_questions.items()):
        ax = axes_flat[idx]
        
        # Count responses
        response_counts = evaluations_df[column].value_counts()
        
        # Ensure all options are represented (even if count is 0)
        response_counts = pd.Series(index=question_info['options'], data=0)
        actual_counts = evaluations_df[column].value_counts()
        response_counts.update(actual_counts)
        response_counts = response_counts.reindex(question_info['options'])

        # Create color gradient for this subplot
        num_options = len(question_info['options'])
        base_color = base_colors[column]
        # Convert base color to RGB
        base_rgb = plt.matplotlib.colors.to_rgb(base_color)
        # Create gradient by adjusting alpha
        colors = [plt.matplotlib.colors.to_rgba(base_color, alpha=0.4 + (0.6 * i/(num_options-1))) 
                 for i in range(num_options)]

        # Create bar plot with color gradient
        bars = ax.bar(range(len(response_counts)), response_counts.values, color=colors)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only show label if there are responses
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'n={int(height)}',
                        ha='center', va='bottom')

        # Customize the plot
        ax.set_title(question_info['title'], pad=20, fontsize=14)
        ax.set_xticks(range(len(question_info['options'])))
        ax.set_xticklabels(question_info['options'], rotation=45, ha='right')
        ax.set_ylabel('Number of Responses')
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save the figure with high quality
    plt.savefig(f'{output_dir}/feedback_analysis.svg', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_sus_responses(evaluations_df, output_dir):
    """Create a comprehensive visualization of SUS responses"""
    if evaluations_df.empty:
        return

    # Define SUS questions
    sus_questions = {
        'sus_q1': "I think that I would like to use this system frequently.",
        'sus_q2': "I found the system unnecessarily complex.",
        'sus_q3': "I thought the system was easy to use.",
        'sus_q4': "I think that I would need the support of a technical person to be able to use this system.",
        'sus_q5': "I found the various functions in this system were well integrated.",
        'sus_q6': "I thought there was too much inconsistency in this system.",
        'sus_q7': "I would imagine that most people would learn to use this system very quickly.",
        'sus_q8': "I found the system very cumbersome to use.",
        'sus_q9': "I felt very confident using the system.",
        'sus_q10': "I needed to learn a lot of things before I could get going with this system."
    }

    # Response options and their colors
    response_options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # Red, Orange, Green, Blue, Purple

    # Process responses
    response_data = []
    
    for q_id, question in sus_questions.items():
        # Get responses for this question
        q_responses = []
        for _, row in evaluations_df.iterrows():
            if pd.notna(row['sus_responses']):
                responses = row['sus_responses'] if isinstance(row['sus_responses'], dict) else json.loads(row['sus_responses'])
                if q_id in responses:
                    q_responses.append(responses[q_id])
        
        # Count responses
        response_counts = pd.Series(q_responses).value_counts()
        
        # Create a row with counts for each option (fill with 0 if no responses)
        row_data = [response_counts.get(opt, 0) for opt in response_options]
        response_data.append(row_data)

    # Convert to numpy array for easier manipulation
    response_data = np.array(response_data)
    
    # Calculate percentages
    totals = response_data.sum(axis=1, keepdims=True)
    percentages = (response_data / totals) * 100

    # Create the visualization
    fig, ax = plt.subplots(figsize=(15, 14))  # Increased height from 12 to 14

    # Plot stacked horizontal bars - Reverse the order of questions
    left = np.zeros(len(sus_questions))
    bars = []
    for i, (option, color) in enumerate(zip(response_options, colors)):
        bars.append(ax.barh(range(len(sus_questions)-1, -1, -1),  # Reversed range
                          percentages[:, i], 
                          left=left, 
                          color=color, 
                          label=option))
        left += percentages[:, i]

    # Customize the plot with larger fonts and more spacing
    ax.set_yticks(range(len(sus_questions)-1, -1, -1))  # Reversed range
    # Create labels with only Q-numbers bold
    labels = [f"\u0466{i+1}. {q}" for i, q in enumerate(sus_questions.values())]  # Using unicode for bold Q
    ax.set_yticklabels([f"$\\mathbf{{Q{i+1}}}$. {q}" for i, q in enumerate(sus_questions.values())],
                       fontsize=13, wrap=True)
    
    # Add more spacing between y-axis labels
    plt.gca().margins(y=0.02)  # Add small margin to prevent text cutoff

    # Add percentage labels on the bars with larger font
    for i in range(len(sus_questions)-1, -1, -1):  # Reversed range
        left = 0
        for j in range(len(response_options)):
            if percentages[i, j] > 0:  # Only show label if there are responses
                width = percentages[i, j]
                count = response_data[i, j]
                ax.text(left + width/2, len(sus_questions)-1-i,  # Adjusted y-position
                       f'{width:.0f}%\n(n={count})',
                       ha='center', va='center',
                       fontsize=15)
                left += width

    # Add title and labels with larger fonts
    plt.title('System Usability Scale (SUS) Response Distribution', pad=40, fontsize=16)
    plt.xlabel('Percentage of Responses', fontsize=14)

    # Add legend with larger font and size
    legend = ax.legend(title="Responses", bbox_to_anchor=(0.5, 1.15), loc='center', ncol=5, 
             fontsize=14, title_fontsize=16)
    # Make legend larger overall
    legend.set_bbox_to_anchor((0.5, 1.15, 0, 0))  # x, y, width, height
    plt.setp(legend.get_patches(), linewidth=2)  # Make color boxes bigger
    
    # Increase spacing between legend items
    legend._legend_box.sep = 15  # Pixels between legend items
    legend._legend_box.pad = 10  # Pixels around the legend

    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Ensure x-axis goes from 0 to 100
    ax.set_xlim(0, 100)

    # Make x-axis tick labels larger
    ax.tick_params(axis='x', labelsize=12)

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_dir}/sus_responses.svg', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Main function to run analysis"""
    print("Fetching data from Supabase...")
    evaluations_df, answers_df = fetch_data()
    
    print(f"Retrieved {len(evaluations_df)} evaluation records and {len(answers_df)} answer records")
    
    # Generate metrics
    print("Generating metrics...")
    metrics = generate_metrics(evaluations_df, answers_df)
    
    # Save metrics to file
    os.makedirs('User_Evaluation_Analysis', exist_ok=True)
    with open('User_Evaluation_Analysis/metrics_results.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print("Metrics saved to User_Evaluation_Analysis/metrics_results.json")
    
    # Visualize metrics
    print("Creating visualizations...")
    visualize_metrics(metrics)
    
    # Create feedback visualization
    print("Creating feedback analysis visualization...")
    visualize_feedback_responses(evaluations_df, 'User_Evaluation_Analysis')
    
    # Create SUS responses visualization
    print("Creating SUS responses visualization...")
    visualize_sus_responses(evaluations_df, 'User_Evaluation_Analysis')
    
    print("Analysis complete! Visualizations saved to 'User_Evaluation_Analysis' directory")

if __name__ == "__main__":
    main() 