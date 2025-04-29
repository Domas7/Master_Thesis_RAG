from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def evaluate_generation(rag_model, test_questions, reference_answers, model_name="openai"):
    """
    Evaluate the generation component of the RAG system.
    
    Args:
        rag_model: Your RAG model instance
        test_questions: List of test questions
        reference_answers: Dictionary mapping questions to reference answers
        model_name: Which model to use for generation
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        'bert_scores': [],
        'semantic_similarity': [],
        'answer_relevance': [],
        'factual_accuracy': [],
        'groundedness': []
    }
    
    # Initialize MPNet model for both semantic similarity and BERTScore
    sentence_model_name = 'sentence-transformers/all-mpnet-base-v2'
    sentence_model = SentenceTransformer(sentence_model_name)
    
    for question in test_questions:
        print(f"Evaluating generation for question: {question} with {model_name}")
        
        # Generate answer
        start_time = time.time()
        generated_answer = rag_model.query(question, model_name)
        generation_time = time.time() - start_time
        print(f"  Generation time: {generation_time:.2f} seconds")
        
        # Remove the sources section for evaluation
        if "4. Sources:" in generated_answer:
            generated_answer = generated_answer.split("4. Sources:")[0].strip()
        
        # Get reference answer
        reference_answer = reference_answers.get(question, "")
        
        # Calculate BERTScore using the same model as semantic similarity
        try:
            # Use the same model for BERTScore
            P, R, F1 = bert_score([generated_answer], [reference_answer], 
                                  lang="en", 
                                  rescale_with_baseline=False,
                                  model_type="microsoft/mpnet-base",
                                  num_layers=11) # (default is 8)
            
            print(f"  BERTScore raw values - P: {P.item():.2f}, R: {R.item():.2f}, F1: {F1.item():.2f}")
            
            # Apply additional normalization if needed
            bert_precision = max(0, min(1, P.item()))
            bert_recall = max(0, min(1, R.item()))
            bert_f1 = max(0, min(1, F1.item()))
            
        except Exception as e:
            print(f"  Error calculating BERTScore: {e}")
            bert_precision, bert_recall, bert_f1 = 0.0, 0.0, 0.0
        
        # Calculate semantic similarity using the same model
        gen_embedding = sentence_model.encode(generated_answer)
        ref_embedding = sentence_model.encode(reference_answer)
        similarity = cosine_similarity([gen_embedding], [ref_embedding])[0][0]
        
        # Get retrieved documents for groundedness evaluation.
        retrieved_docs = rag_model.retriever.get_relevant_documents(question)
        
        # For answer relevance, factual accuracy, and groundedness, using LLM-as-a-judge
        try:
            relevance, accuracy, groundedness = evaluate_with_llm(
                question, generated_answer, reference_answer, retrieved_docs
            )
        except Exception as e:
            print(f"  Error in LLM evaluation: {e}")
            relevance, accuracy, groundedness = 0.5, 0.5, 0.5
        
        # Store results
        results['bert_scores'].append({
            'precision': bert_precision,
            'recall': bert_recall,
            'f1': bert_f1
        })
        results['semantic_similarity'].append(similarity)
        results['answer_relevance'].append(relevance)
        results['factual_accuracy'].append(accuracy)
        results['groundedness'].append(groundedness)
        
        print(f"  BERTScore F1: {bert_f1:.2f}, Similarity: {similarity:.2f}")
        print(f"  Relevance: {relevance:.2f}, Accuracy: {accuracy:.2f}, Groundedness: {groundedness:.2f}")
    
    # Average the results
    final_results = {}
    for metric in results:
        if metric == 'bert_scores':
            final_results['bert_precision'] = sum(s['precision'] for s in results[metric]) / len(results[metric])
            final_results['bert_recall'] = sum(s['recall'] for s in results[metric]) / len(results[metric])
            final_results['bert_f1'] = sum(s['f1'] for s in results[metric]) / len(results[metric])
        else:
            final_results[metric] = sum(results[metric]) / len(results[metric])
    
    return final_results

# remember to cite the code reference later https://lightning.ai/panchamsnotes/studios/evaluate-your-rag-part-2-llm-as-a-judge?section=featured
def evaluate_with_llm(question, generated_answer, reference_answer, retrieved_docs):
    """
    Use an LLM to evaluate answer relevance, factual accuracy, and groundedness.
    
    Returns:
        Tuple of (relevance_score, accuracy_score, groundedness_score) between 0 and 1
    """
    # Use GPT-4o-mini for evaluation
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    evaluation_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator for question-answering systems. Please evaluate the following answer based on three criteria:
    
    Question: {question}
    
    Generated Answer: {generated_answer}
    
    Reference Answer: {reference_answer}
    
    Retrieved Context: {context}
    
    Please evaluate on a scale of 0 to 10 for each criterion:
    
    1. Relevance: How relevant is the generated answer to the question?
    2. Factual Accuracy: How factually accurate is the generated answer compared to the reference?
    3. Groundedness: How well is the generated answer grounded in the retrieved context?
    
    Format your response as three numbers separated by commas, e.g., "7,8,6"
    """)
    
    result = llm.invoke(evaluation_prompt.format(
        question=question,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        context=context
    ))
    
    try:
        scores = result.content.strip().split(',')
        relevance = float(scores[0]) / 10
        accuracy = float(scores[1]) / 10
        groundedness = float(scores[2]) / 10
        return relevance, accuracy, groundedness
    except:
        # Fallback if parsing fails
        return 0.0, 0.0, 0.0

def run_bertscore_sanity_check():
    """Run a sanity check to see what BERTScore returns for unrelated texts."""
    from bert_score import score as bert_score
    
    unrelated_pairs = [
        ("The sky is blue and the sun is shining.", "The sky is blue and the sun is shining."),
        ("abra kadabra o dæven det blir knallbra", "Photosynthesis is the process by which plants convert sunlight to energy."),
        ("The algorithm complexity is O(n log n).", "The Eiffel Tower is located in Paris, France.")
    ]
    
    # Change the model to a more compatible one
    for text1, text2 in unrelated_pairs:
        P, R, F1 = bert_score([text1], [text2], lang="en", model_type="microsoft/mpnet-base", num_layers=11)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"BERTScore - P: {P.item():.2f}, R: {R.item():.2f}, F1: {F1.item():.2f}\n")

# Run the sanity check to understand BERTScore behavior
run_bertscore_sanity_check()
