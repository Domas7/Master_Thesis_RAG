import json
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.evaluation import load_evaluator
from rouge_score import rouge_scorer

class RAGEvaluator:
    def __init__(self, test_database_path: str):
        """Initialize the RAG evaluator with test database."""
        # Load test database
        with open(test_database_path, 'r') as f:
            self.test_data = json.load(f)
        
        # Initialize models and scorers
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_retrieval(self, 
                          queries: List[str], 
                          retrieved_chunks_openai: List[List[str]], 
                          retrieved_chunks_llama: List[List[str]]) -> Dict:
        """Evaluate retrieval performance for both models."""
        results = {
            'openai': {'precision': [], 'recall': [], 'mrr': [], 'hit_rate': [], 'ndcg': []},
            'llama': {'precision': [], 'recall': [], 'mrr': [], 'hit_rate': [], 'ndcg': []}
        }
        
        for i, query in enumerate(queries):
            # Get ground truth chunks for this query
            gt_chunks = self.test_data[i]['relevant_chunk_ids']
            
            # Evaluate OpenAI retrieval
            openai_metrics = self._calculate_retrieval_metrics(
                gt_chunks, retrieved_chunks_openai[i]
            )
            for metric, value in openai_metrics.items():
                results['openai'][metric].append(value)
            
            # Evaluate Llama retrieval
            llama_metrics = self._calculate_retrieval_metrics(
                gt_chunks, retrieved_chunks_llama[i]
            )
            for metric, value in llama_metrics.items():
                results['llama'][metric].append(value)
        
        # Average the results
        for system in results:
            for metric in results[system]:
                results[system][metric] = np.mean(results[system][metric])
        
        return results

    def evaluate_answer_quality(self, 
                              generated_answers: List[str], 
                              model_name: str) -> Dict:
        """Evaluate the quality of generated answers."""
        results = {
            'rouge_scores': [],
            'semantic_similarity': [],
            'answer_relevance': [],
            'factual_accuracy': []
        }
        
        for i, answer in enumerate(generated_answers):
            expected_answer = self.test_data[i]['expected_answer']
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(expected_answer, answer)
            results['rouge_scores'].append({
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            })
            
            # Calculate semantic similarity
            answer_embedding = self.embedding_model.encode([answer])[0]
            expected_embedding = self.embedding_model.encode([expected_answer])[0]
            similarity = cosine_similarity([answer_embedding], [expected_embedding])[0][0]
            results['semantic_similarity'].append(similarity)
            
            # Calculate answer relevance and factual accuracy
            # (This could be done using an LLM-as-judge approach)
            relevance, accuracy = self._evaluate_with_llm(
                answer, 
                expected_answer, 
                self.test_data[i]['question']
            )
            results['answer_relevance'].append(relevance)
            results['factual_accuracy'].append(accuracy)
        
        # Average the results
        final_results = {}
        for metric in results:
            if metric == 'rouge_scores':
                final_results['rouge1'] = np.mean([s['rouge1'] for s in results[metric]])
                final_results['rouge2'] = np.mean([s['rouge2'] for s in results[metric]])
                final_results['rougeL'] = np.mean([s['rougeL'] for s in results[metric]])
            else:
                final_results[metric] = np.mean(results[metric])
        
        return final_results

    def _calculate_retrieval_metrics(self, 
                                   gt_chunks: List[str], 
                                   retrieved_chunks: List[str]) -> Dict:
        """Calculate retrieval metrics for a single query."""
        # Calculate precision
        precision = len(set(retrieved_chunks) & set(gt_chunks)) / len(retrieved_chunks)
        
        # Calculate recall
        recall = len(set(retrieved_chunks) & set(gt_chunks)) / len(gt_chunks)
        
        # Calculate MRR
        mrr = 0
        for i, chunk in enumerate(retrieved_chunks):
            if chunk in gt_chunks:
                mrr = 1.0 / (i + 1)
                break
        
        # Calculate hit rate
        hit_rate = 1 if len(set(retrieved_chunks) & set(gt_chunks)) > 0 else 0
        
        # Calculate NDCG
        ndcg = self._calculate_ndcg(gt_chunks, retrieved_chunks)
        
        return {
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'hit_rate': hit_rate,
            'ndcg': ndcg
        }

    def _calculate_ndcg(self, 
                       gt_chunks: List[str], 
                       retrieved_chunks: List[str], 
                       k: int = None) -> float:
        """Calculate NDCG for a single query."""
        if k is None:
            k = len(retrieved_chunks)
        
        dcg = 0
        idcg = 0
        
        # Calculate DCG
        for i, chunk in enumerate(retrieved_chunks[:k]):
            rel = 1 if chunk in gt_chunks else 0
            dcg += rel / np.log2(i + 2)
        
        # Calculate IDCG
        for i in range(min(len(gt_chunks), k)):
            idcg += 1 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0

# Initialize evaluator
evaluator = RAGEvaluator('TEST_DATABASE.json')

# Example usage with your RAG systems
queries = [item['question'] for item in evaluator.test_data]
retrieved_chunks_openai = []  
retrieved_chunks_llama = []   
generated_answers_openai = [] 
generated_answers_llama = [] 

# Evaluate retrieval
retrieval_results = evaluator.evaluate_retrieval(
    queries,
    retrieved_chunks_openai,
    retrieved_chunks_llama
)

# Evaluate answer quality
openai_answer_quality = evaluator.evaluate_answer_quality(
    generated_answers_openai,
    'openai'
)
llama_answer_quality = evaluator.evaluate_answer_quality(
    generated_answers_llama,
    'llama'
)

# Print results
print("Retrieval Results:")
print(json.dumps(retrieval_results, indent=2))
print("\nOpenAI Answer Quality:")
print(json.dumps(openai_answer_quality, indent=2))
print("\nLlama Answer Quality:")
print(json.dumps(llama_answer_quality, indent=2))