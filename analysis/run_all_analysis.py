#!/usr/bin/env python3
import os
import subprocess
import json
import shutil
from datetime import datetime
import pandas as pd

def run_analysis():
    """Run all analysis scripts and generate visualizations"""
    print("=" * 80)
    print("MASTER THESIS RAG APPLICATION ANALYSIS")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Create output directory
    report_dir = "User_Evaluation_Analysis"
    
    # Clean up existing report directory to avoid duplicates
    if os.path.exists(report_dir):
        print(f"Cleaning up existing report directory: {report_dir}")
        shutil.rmtree(report_dir)
    
    # Create fresh directory
    os.makedirs(report_dir, exist_ok=True)
    
    # Run analysis scripts
    print("\nRunning general evaluation metrics analysis...")
    subprocess.run(["python", "user_evaluation_metrics.py"], check=True)
    
    print("\nRunning model comparison analysis...")
    subprocess.run(["python", "model_comparison_metrics.py"], check=True)
    
    print("\nRunning SUS score analysis...")
    subprocess.run(["python", "sus_analysis.py"], check=True)
    
    print("\n" + "=" * 80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Visualizations saved to: {report_dir}")
    print("=" * 80)

def compile_report(report_dir):
    """Compile all analysis results into a comprehensive report"""
    print("\nCompiling comprehensive analysis report...")
    
    report = {
        "title": "Master Thesis RAG Application User Evaluation Analysis",
        "generated_at": datetime.now().isoformat(),
        "sections": []
    }
    
    # Function to safely load JSON
    def load_json_file(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    # 1. Add general metrics
    metrics_file = f"{report_dir}/metrics_results.json"
    if os.path.exists(metrics_file):
        metrics_data = load_json_file(metrics_file)
        if metrics_data:
            report["sections"].append({
                "title": "General Evaluation Metrics",
                "data": metrics_data
            })
            
            # Copy visualization files - only task_difficulty and task completion times
            if os.path.exists("visualization"):
                os.system(f"mkdir -p {report_dir}/visualization")
                os.system(f"cp visualization/task_difficulty.svg {report_dir}/visualization/")
                os.system(f"cp visualization/task_completion_times.svg {report_dir}/visualization/")
    
    # 2. Add model comparison metrics
    model_comparison_file = f"{report_dir}/model_comparison_summary.json"
    if os.path.exists(model_comparison_file):
        model_data = load_json_file(model_comparison_file)
        if model_data:
            report["sections"].append({
                "title": "Model Comparison Analysis",
                "data": model_data
            })
            
    # 3. Add SUS analysis
    sus_file = f"{report_dir}/sus_summary.json"
    if os.path.exists(sus_file):
        sus_data = load_json_file(sus_file)
        if sus_data:
            report["sections"].append({
                "title": "System Usability Scale (SUS) Analysis",
                "data": sus_data
            })
    
    # Save the comprehensive report
    with open(f"{report_dir}/comprehensive_analysis.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate a human-readable summary report
    generate_human_readable_report(report, report_dir)

def generate_human_readable_report(report_data, report_dir):
    """Generate a human-readable report in markdown format"""
    markdown = f"# {report_data['title']}\n\n"
    markdown += f"*Generated: {report_data['generated_at']}*\n\n"
    markdown += "## Executive Summary\n\n"
    
    # Extract key metrics from each section for the executive summary
    task_completion_rate = None
    avg_sus_score = None
    model_comparison = None
    
    for section in report_data['sections']:
        if section['title'] == "General Evaluation Metrics" and "task_completion" in section['data']:
            task_data = section['data']['task_completion']
            task_completion_rate = task_data.get('success_rate')
            
        elif section['title'] == "System Usability Scale (SUS) Analysis":
            avg_sus_score = section['data'].get('mean_score')
            sus_interpretation = section['data'].get('average_interpretation')
            
        elif section['title'] == "Model Comparison Analysis" and "success_metrics" in section['data']:
            model_comparison = section['data']['success_metrics']
    
    # Add executive summary content
    if task_completion_rate is not None:
        markdown += f"- Overall task completion rate: **{task_completion_rate:.1f}%**\n"
    
    if avg_sus_score is not None:
        markdown += f"- System Usability Scale (SUS) score: **{avg_sus_score:.1f}** ({sus_interpretation})\n"
    
    if model_comparison:
        markdown += "- Model performance comparison:\n"
        for model, metrics in model_comparison.items():
            if model == "Success Rate" and isinstance(metrics, dict):
                for model_name, success_rate in metrics.items():
                    markdown += f"  - {model_name}: **{success_rate:.1f}%** success rate\n"
    
    # Add detailed sections
    for section in report_data['sections']:
        markdown += f"\n## {section['title']}\n\n"
        
        if section['title'] == "General Evaluation Metrics":
            # Task statistics
            if "task_stats" in section['data']:
                markdown += "### Task Performance\n\n"
                markdown += "| Task | Mean Time (min) | Min Time (min) | Max Time (min) | Success Rate (%) |\n"
                markdown += "|------|----------------|----------------|----------------|------------------|\n"
                
                for task in section['data']['task_stats']:
                    # Check if min_time and max_time exist in the data
                    min_time = task.get('min_time', 'N/A')
                    max_time = task.get('max_time', 'N/A')
                    
                    # Format the values appropriately
                    min_time_str = f"{min_time:.2f}" if isinstance(min_time, (int, float)) else min_time
                    max_time_str = f"{max_time:.2f}" if isinstance(max_time, (int, float)) else max_time
                    
                    markdown += f"| {task['task_id']} | {task['mean_time']:.2f} | {min_time_str} | {max_time_str} | {task['success_rate']:.1f}% |\n"
                
                # Add task completion time visualization if available
                if os.path.exists(f"{report_dir}/visualization/task_completion_times.svg"):
                    markdown += "\n![Task Completion Times](visualization/task_completion_times.svg)\n"
            
            # Task difficulty
            if "task_difficulty" in section['data']:
                markdown += "\n### Task Difficulty Perception\n\n"
                markdown += "*Percentage of users rating task difficulty:*\n\n"
                
                # Task difficulty visualization 
                markdown += "![Task Difficulty Distribution](visualization/task_difficulty.svg)\n"
        
        elif section['title'] == "Model Comparison Analysis":
            markdown += "### Model Performance Comparison\n\n"
            
            # Check if we have time metrics
            if "time_metrics" in section['data']:
                markdown += "#### Completion Time\n\n"
                markdown += "| Model | Mean Time (min) | Median Time (min) | Sample Count |\n"
                markdown += "|-------|----------------|-------------------|-------------|\n"
                
                for model, metrics in section['data']['time_metrics'].items():
                    if model == "Mean Time (min)" and isinstance(metrics, dict):
                        for model_name, mean_time in metrics.items():
                            median_time = section['data']['time_metrics']['Median Time (min)'][model_name]
                            sample_count = section['data']['time_metrics']['Sample Count'][model_name]
                            markdown += f"| {model_name} | {mean_time:.2f} | {median_time:.2f} | {sample_count} |\n"
            
            # Add visualization references - only the ones we kept
            markdown += "\n![Time to Complete Tasks by Model](model_comparison/model_time_comparison.png)\n\n"
            markdown += "![Average Number of Attempts by Task and Model](model_comparison/task_model_attempts.png)\n"
        
        elif section['title'] == "System Usability Scale (SUS) Analysis":
            markdown += f"### Overall SUS Score: {section['data']['mean_score']:.1f} - {section['data']['average_interpretation']}\n\n"
            markdown += f"- Median: {section['data']['median_score']:.1f}\n"
            markdown += f"- Range: {section['data']['min_score']:.1f} to {section['data']['max_score']:.1f}\n"
            markdown += f"- Standard Deviation: {section['data']['std_dev']:.2f}\n"
            markdown += f"- Number of Respondents: {section['data']['num_respondents']}\n\n"
            
            # Add interpretation distribution
            if "interpretation_distribution" in section['data']:
                markdown += "#### SUS Score Interpretation\n\n"
                markdown += "| Grade | Count | Percentage |\n"
                markdown += "|-------|-------|------------|\n"
                
                total = sum(section['data']['interpretation_distribution'].values())
                for grade, count in section['data']['interpretation_distribution'].items():
                    percentage = (count / total) * 100
                    markdown += f"| {grade} | {count} | {percentage:.1f}% |\n"
            
            # Add visualization references - only the ones we kept
            markdown += "\n![SUS Scores by User](sus_analysis/sus_scores_by_user.png)\n\n"
            markdown += "![Average SUS Scores by Question](sus_analysis/sus_scores_by_question.png)\n\n"
            markdown += "![Distribution of SUS Score Grades](sus_analysis/sus_grade_distribution.png)\n"
    
    # Add conclusion
    markdown += "\n## Conclusion\n\n"
    markdown += "This analysis provides valuable insights into the usability and effectiveness of the RAG application. "
    
    if avg_sus_score is not None:
        if avg_sus_score >= 68:
            markdown += f"With an average SUS score of {avg_sus_score:.1f}, the system demonstrates good usability above the industry average (68). "
        else:
            markdown += f"With an average SUS score of {avg_sus_score:.1f}, which is below the industry average (68), there are opportunities for improving usability. "
    
    if task_completion_rate is not None:
        if task_completion_rate >= 80:
            markdown += f"The high task completion rate ({task_completion_rate:.1f}%) indicates that users were generally successful in using the application to complete the assigned tasks."
        elif task_completion_rate >= 50:
            markdown += f"The moderate task completion rate ({task_completion_rate:.1f}%) suggests that while many users were able to complete tasks, there may be room for improvement in task design or system responsiveness."
        else:
            markdown += f"The low task completion rate ({task_completion_rate:.1f}%) suggests significant challenges in task design or system usability that should be addressed in future iterations."
    
    # Save markdown report
    with open(f"{report_dir}/analysis_summary.md", 'w') as f:
        f.write(markdown)
    
    print(f"Human-readable summary report saved to: {report_dir}/analysis_summary.md")

if __name__ == "__main__":
    run_analysis() 