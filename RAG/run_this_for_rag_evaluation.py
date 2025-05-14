import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import sys

# Add the parent directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.chatBot.rag_models import get_rag_model
from RAG.retrieval_evaluation import evaluate_retrieval
from RAG.generation_evaluation import evaluate_generation

def main():
    # Initialize your RAG model
    print("Initializing RAG model...")
    
    rag_model = get_rag_model()
    
    # Load test dataset
    print("Loading test dataset...")
    test_db_path = Path(__file__).parent.parent / "TEST_DATABASE_2_WRONG.json"
    with open(test_db_path, 'r') as f:
        test_data = json.load(f)
    
    # Extract test questions and ground truth
    test_questions = [item['question'] for item in test_data]
    reference_answers = {item['question']: item['expected_answer'] for item in test_data}
    ground_truth_docs = {item['question']: item['relevant_chunk_ids'] for item in test_data}
    
    # Create results directory
    results_dir = Path(__file__).parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate retrieval
    print("\n=== Evaluating Retrieval Component ===")
    try:
        retrieval_results = evaluate_retrieval(rag_model, test_questions, ground_truth_docs)
        
        # Save retrieval results
        with open(results_dir / "retrieval_results.json", 'w') as f:
            json.dump(retrieval_results, f, indent=2)
        
        print("Retrieval Results:")
        print(json.dumps(retrieval_results, indent=2))
    except Exception as e:
        print(f"Error in retrieval evaluation: {e}")
        retrieval_results = {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "error": str(e)
        }
    
    # Evaluate generation for each model
    model_names = ["openai", "llama"]
    generation_results = {}
    
    # Ask user how many questions to test for generation
    num_questions = 5  # Default
    try:
        user_input = input(f"\nHow many questions to test for generation? (default: {num_questions}, max: {len(test_questions)}): ")
        if user_input.strip():
            num_questions = min(int(user_input), len(test_questions))
    except ValueError:
        print(f"Using default: {num_questions} questions")
    
    for model_name in model_names:
        print(f"\n=== Evaluating Generation Component ({model_name}) ===")
        print(f"Testing with {num_questions} questions...")
        
        generation_results[model_name] = evaluate_generation(
            rag_model, test_questions[:num_questions], reference_answers, model_name
        )
        
        # Save generation results
        with open(results_dir / f"generation_results_{model_name}.json", 'w') as f:
            json.dump(generation_results[model_name], f, indent=2)
        
        print(f"Generation Results ({model_name}):")
        print(json.dumps(generation_results[model_name], indent=2))
    
    # Save combined results
    combined_results = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_test_questions": len(test_questions),
            "num_generation_questions": num_questions
        }
    }
    
    with open(results_dir / "combined_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nAll evaluation results saved to {results_dir}")
    
    # Generate summary report
    generate_summary_report(combined_results, results_dir)

def generate_summary_report(results, results_dir):
    """Generate a human-readable summary report of the evaluation results"""
    report = []
    report.append("# RAG System Evaluation Summary")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Retrieval metrics
    report.append("## Retrieval Performance")
    retrieval = results["retrieval"]
    report.append(f"- Precision: {retrieval['precision']:.4f}")
    report.append(f"- Recall: {retrieval['recall']:.4f}")
    report.append(f"- MRR (Mean Reciprocal Rank): {retrieval['mrr']:.4f}")
    report.append(f"- Hit Rate: {retrieval['hit_rate']:.4f}")
    report.append(f"- NDCG: {retrieval['ndcg']:.4f}")
    report.append("")
    
    # Generation metrics
    report.append("## Generation Performance")
    for model_name, gen_results in results["generation"].items():
        report.append(f"### {model_name.upper()}")
        
        report.append("#### Semantic Metrics")
        report.append(f"- BERTScore Precision: {gen_results['bert_precision']:.4f}")
        report.append(f"- BERTScore Recall: {gen_results['bert_recall']:.4f}")
        report.append(f"- BERTScore F1: {gen_results['bert_f1']:.4f}")
        report.append(f"- Semantic Similarity: {gen_results['semantic_similarity']:.4f}")
        report.append("")
        
        report.append("#### ROUGE Metrics")
        report.append(f"- ROUGE-1 F1: {gen_results['rouge1']:.4f}")
        report.append(f"- ROUGE-2 F1: {gen_results['rouge2']:.4f}")
        report.append(f"- ROUGE-L F1: {gen_results['rougeL']:.4f}")
        report.append("")
        
        report.append("#### LLM-as-judge Metrics (custom implementation)")
        report.append(f"- Answer Relevance: {gen_results['answer_relevance']:.4f}")
        report.append(f"- Factual Accuracy: {gen_results['factual_accuracy']:.4f}")
        report.append(f"- Groundedness: {gen_results['groundedness']:.4f}")
        
        # Add GEval scores if available
        if 'geval_score' in gen_results:
            report.append("")
            report.append("#### GEval Metrics (LLM-as-judge using deepeval)")
            report.append(f"- GEval Correctness: {gen_results['geval_score']:.4f}")
            
            # Add the comparison metrics
            if 'geval_relevance' in gen_results:
                report.append(f"- GEval Relevance: {gen_results['geval_relevance']:.4f}")
            if 'geval_accuracy' in gen_results:
                report.append(f"- GEval Accuracy: {gen_results['geval_accuracy']:.4f}")
            if 'geval_groundedness' in gen_results:
                report.append(f"- GEval Groundedness: {gen_results['geval_groundedness']:.4f}")
                
            # Add comparison summary
            report.append("")
            report.append("#### Comparison between LLM evaluation methods")
            if 'geval_relevance' in gen_results and 'answer_relevance' in gen_results:
                diff = gen_results['geval_relevance'] - gen_results['answer_relevance']
                report.append(f"- Relevance difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            if 'geval_accuracy' in gen_results and 'factual_accuracy' in gen_results:
                diff = gen_results['geval_accuracy'] - gen_results['factual_accuracy']
                report.append(f"- Accuracy difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            if 'geval_groundedness' in gen_results and 'groundedness' in gen_results:
                diff = gen_results['geval_groundedness'] - gen_results['groundedness']
                report.append(f"- Groundedness difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            
        report.append("")
    
    # Write report to file
    with open(results_dir / "evaluation_summary.md", 'w') as f:
        f.write("\n".join(report))
    
    print(f"Summary report generated: {results_dir / 'evaluation_summary.md'}")

if __name__ == "__main__":
    main()