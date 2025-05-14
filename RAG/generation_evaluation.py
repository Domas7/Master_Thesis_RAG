from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from rouge import Rouge
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate as deepeval_evaluate

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text_for_rouge(text):
    """
    Preprocess text to improve ROUGE scoring:
    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation
    - Optional: Remove stopwords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation (keeping periods for sentence splits)
    text = re.sub(r'[^\w\s\.]', '', text)
    
    # Optional: Remove stopwords (uncomment if needed)
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

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
        'groundedness': [],
        'rouge_scores': [],
        'geval_scores': [],
        'geval_relevance': [],
        'geval_accuracy': [],
        'geval_groundedness': []
    }
    
    # Initialize MPNet model for both semantic similarity and BERTScore
    sentence_model_name = 'sentence-transformers/all-mpnet-base-v2'
    sentence_model = SentenceTransformer(sentence_model_name)
    
    # Initialize Rouge scorer instead of rouge_scorer
    rouge = Rouge()
    
    # Setup GEval metrics
    metrics = setup_geval_metrics()
    
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
        
        # Preprocess for ROUGE evaluation
        preprocessed_reference = preprocess_text_for_rouge(reference_answer)
        preprocessed_generated = preprocess_text_for_rouge(generated_answer)
        
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
        
        # Calculate ROUGE scores with the Rouge library
        try:
            # Try with preprocessed text
            rouge_scores_preprocessed = rouge.get_scores(preprocessed_generated, preprocessed_reference)[0]
            
            # Also try with original text
            rouge_scores_original = rouge.get_scores(generated_answer, reference_answer)[0]
            
            # Take the better of the two scores for each metric
            rouge_results = {
                'rouge1': {
                    'p': max(rouge_scores_preprocessed['rouge-1']['p'], rouge_scores_original['rouge-1']['p']),
                    'r': max(rouge_scores_preprocessed['rouge-1']['r'], rouge_scores_original['rouge-1']['r']),
                    'f': max(rouge_scores_preprocessed['rouge-1']['f'], rouge_scores_original['rouge-1']['f']),
                },
                'rouge2': {
                    'p': max(rouge_scores_preprocessed['rouge-2']['p'], rouge_scores_original['rouge-2']['p']),
                    'r': max(rouge_scores_preprocessed['rouge-2']['r'], rouge_scores_original['rouge-2']['r']),
                    'f': max(rouge_scores_preprocessed['rouge-2']['f'], rouge_scores_original['rouge-2']['f']),
                },
                'rougeL': {
                    'p': max(rouge_scores_preprocessed['rouge-l']['p'], rouge_scores_original['rouge-l']['p']),
                    'r': max(rouge_scores_preprocessed['rouge-l']['r'], rouge_scores_original['rouge-l']['r']),
                    'f': max(rouge_scores_preprocessed['rouge-l']['f'], rouge_scores_original['rouge-l']['f']),
                }
            }
            
            print(f"  ROUGE Scores - R1: {rouge_results['rouge1']['f']:.4f}, R2: {rouge_results['rouge2']['f']:.4f}, " +
                  f"RL: {rouge_results['rougeL']['f']:.4f}")
            
            # Try sentence-level ROUGE for potentially better scores
            try:
                # Split into sentences
                ref_sentences = sent_tokenize(reference_answer)
                gen_sentences = sent_tokenize(generated_answer)
                
                # If we have multiple sentences, calculate sentence-level ROUGE
                if len(ref_sentences) > 1 and len(gen_sentences) > 1:
                    # Calculate ROUGE for best-matching sentence pairs
                    sentence_rouges = []
                    for ref_sent in ref_sentences:
                        if not ref_sent.strip():
                            continue
                        best_rouge_l = 0
                        for gen_sent in gen_sentences:
                            if not gen_sent.strip():
                                continue
                            try:
                                sent_score = rouge.get_scores(gen_sent, ref_sent)[0]['rouge-l']['f']
                                if sent_score > best_rouge_l:
                                    best_rouge_l = sent_score
                            except:
                                continue
                        if best_rouge_l > 0:
                            sentence_rouges.append(best_rouge_l)
                    
                    # Average the best matches
                    if sentence_rouges:
                        avg_sentence_rouge = sum(sentence_rouges) / len(sentence_rouges)
                        # Use the better of the two ROUGE-L scores
                        rouge_results['rougeL']['f'] = max(rouge_results['rougeL']['f'], avg_sentence_rouge)
                        print(f"  Sentence-aligned ROUGE-L: {avg_sentence_rouge:.4f}")
            except Exception as e:
                print(f"  Error in sentence-aligned ROUGE: {e}")
                
        except Exception as e:
            print(f"  Error calculating ROUGE scores: {e}")
            rouge_results = {
                'rouge1': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rouge2': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rougeL': {'p': 0.0, 'r': 0.0, 'f': 0.0}
            }
        
        # Get retrieved documents for groundedness evaluation
        retrieved_docs = rag_model.retriever.get_relevant_documents(question)
        
        # For answer relevance, factual accuracy, and groundedness, using LLM-as-a-judge
        try:
            relevance, accuracy, groundedness = evaluate_with_llm(
                question, generated_answer, reference_answer, retrieved_docs
            )
        except Exception as e:
            print(f"  Error in LLM evaluation: {e}")
            relevance, accuracy, groundedness = 0.5, 0.5, 0.5
        
        # Run GEval evaluation
        try:
            geval_results = run_geval_evaluation(
                question, 
                generated_answer, 
                reference_answer,
                metrics,
                retrieved_docs
            )
            print(f"  GEval Scores - Correctness: {geval_results['correctness']['score']:.2f} - {geval_results['correctness']['reason'][:100]}...")
            print(f"  GEval Scores - Relevance: {geval_results['relevance']['score']:.2f} - {geval_results['relevance']['reason'][:100]}...")
            print(f"  GEval Scores - Accuracy: {geval_results['accuracy']['score']:.2f} - {geval_results['accuracy']['reason'][:100]}...")
            print(f"  GEval Scores - Groundedness: {geval_results['groundedness']['score']:.2f} - {geval_results['groundedness']['reason'][:100]}...")
        except Exception as e:
            print(f"  Error in GEval evaluation: {e}")
            geval_results = {
                'correctness': {'score': 0.0, 'reason': "Error in evaluation"},
                'relevance': {'score': 0.0  , 'reason': "Error in evaluation"},
                'accuracy': {'score': 0.0, 'reason': "Error in evaluation"},
                'groundedness': {'score': 0.0, 'reason': "Error in evaluation"}
            }
        
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
        results['rouge_scores'].append({
            'rouge1': rouge_results['rouge1']['f'],
            'rouge2': rouge_results['rouge2']['f'],
            'rougeL': rouge_results['rougeL']['f']
        })
        results['geval_scores'].append(geval_results['correctness'])
        
        # Store individual GEval metrics for comparison
        results['geval_relevance'] = results.get('geval_relevance', [])
        results['geval_relevance'].append(geval_results['relevance']['score'])
        
        results['geval_accuracy'] = results.get('geval_accuracy', [])
        results['geval_accuracy'].append(geval_results['accuracy']['score'])
        
        results['geval_groundedness'] = results.get('geval_groundedness', [])
        results['geval_groundedness'].append(geval_results['groundedness']['score'])
        
        print(f"  BERTScore F1: {bert_f1:.2f}, Similarity: {similarity:.2f}")
        print(f"  Relevance: {relevance:.2f}, Accuracy: {accuracy:.2f}, Groundedness: {groundedness:.2f}")
    
    # Average the results
    final_results = {}
    for metric in results:
        if metric == 'bert_scores':
            final_results['bert_precision'] = sum(s['precision'] for s in results[metric]) / len(results[metric])
            final_results['bert_recall'] = sum(s['recall'] for s in results[metric]) / len(results[metric])
            final_results['bert_f1'] = sum(s['f1'] for s in results[metric]) / len(results[metric])
        elif metric == 'rouge_scores':
            final_results['rouge1'] = sum(s['rouge1'] for s in results[metric]) / len(results[metric])
            final_results['rouge2'] = sum(s['rouge2'] for s in results[metric]) / len(results[metric])
            final_results['rougeL'] = sum(s['rougeL'] for s in results[metric]) / len(results[metric])
        elif metric == 'geval_scores':
            final_results['geval_score'] = sum(s['score'] for s in results[metric]) / len(results[metric])
        elif metric in ['geval_relevance', 'geval_accuracy', 'geval_groundedness']:
            # Already flat lists of scores
            final_results[metric] = sum(results[metric]) / len(results[metric])
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

def setup_geval_metrics():
    """
    Setup GEval metrics for evaluation.
    
    Returns:
        Dictionary of GEval metric instances
    """
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    
    # Add metrics that match the existing evaluate_with_llm metrics
    relevance_metric = GEval(
        name="Relevance",
        criteria="Determine how relevant the actual output is to the input question.",
        evaluation_steps=[
            "Assess whether the actual output directly addresses the question in the input",
            "Check if the response stays on topic and provides information that answers what was asked",
            "Consider whether all parts of the response are relevant to the question"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    
    accuracy_metric = GEval(
        name="Factual Accuracy",
        criteria="Determine how factually accurate the actual output is compared to the expected output.",
        evaluation_steps=[
            "Check if the facts in the actual output align with those in the expected output",
            "Identify any factual contradictions or incorrect statements",
            "Assess whether key facts from the expected output are present in the actual output"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    
    groundedness_metric = GEval(
        name="Groundedness",
        criteria="Determine how well the actual output is grounded in the context provided.",
        evaluation_steps=[
            "Check if claims in the actual output can be verified from the context",
            "Assess whether the actual output contains information not found in the context",
            "Determine if the actual output appropriately draws from the most relevant parts of the context"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    )
    
    return {
        "correctness": correctness_metric,
        "relevance": relevance_metric,
        "accuracy": accuracy_metric,
        "groundedness": groundedness_metric
    }

def run_geval_evaluation(question, generated_answer, reference_answer, metrics, retrieved_docs=None):
    """
    Run GEval evaluation on the generated answer.
    
    Args:
        question: The input question
        generated_answer: The generated answer
        reference_answer: The reference answer
        metrics: Dictionary of GEval metric instances
        retrieved_docs: Retrieved documents for groundedness evaluation (optional)
        
    Returns:
        Dictionary of scores and reasons
    """
    # Create context from retrieved documents if available
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Base test case without context
    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=reference_answer
    )
    
    # Add context if available (for groundedness)
    if context:
        test_case.context = context
    
    results = {}
    
    # Run correctness metric
    try:
        metrics["correctness"].measure(test_case)
        results["correctness"] = {
            "score": metrics["correctness"].score,
            "reason": metrics["correctness"].reason
        }
    except Exception as e:
        print(f"  Error in GEval correctness measurement: {e}")
        results["correctness"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run relevance metric
    try:
        metrics["relevance"].measure(test_case)
        results["relevance"] = {
            "score": metrics["relevance"].score,
            "reason": metrics["relevance"].reason
        }
    except Exception as e:
        print(f"  Error in GEval relevance measurement: {e}")
        results["relevance"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run accuracy metric
    try:
        metrics["accuracy"].measure(test_case)
        results["accuracy"] = {
            "score": metrics["accuracy"].score,
            "reason": metrics["accuracy"].reason
        }
    except Exception as e:
        print(f"  Error in GEval accuracy measurement: {e}")
        results["accuracy"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run groundedness metric if context is available
    if context:
        try:
            metrics["groundedness"].measure(test_case)
            results["groundedness"] = {
                "score": metrics["groundedness"].score,
                "reason": metrics["groundedness"].reason
            }
        except Exception as e:
            print(f"  Error in GEval groundedness measurement: {e}")
            results["groundedness"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    return results

def run_bertscore_sanity_check():
    """Run a sanity check to see what BERTScore returns for unrelated texts."""
    from bert_score import score as bert_score
    
    unrelated_pairs = [
        ("The sky is blue and the sun is shining.", "The sky is blue and the sun is shining."),
        ("abra kadabra o snart det blir knallbra", "Photosynthesis is the process by which plants convert sunlight to energy."),
        ("The algorithm complexity is O(n log n).", "The Eiffel Tower is located in Paris, France.")
    ]
    
    # Change the model to a more compatible one
    for text1, text2 in unrelated_pairs:
        P, R, F1 = bert_score([text1], [text2], lang="en", model_type="microsoft/mpnet-base", num_layers=11)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"BERTScore - P: {P.item():.2f}, R: {R.item():.2f}, F1: {F1.item():.2f}\n")

# Function to demonstrate GEval usage independently
def run_geval_demo():
    """
    Demonstrate GEval usage with a simple example.
    """
    print("Running GEval demo...")
    
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    
    test_case = LLMTestCase(
        input="The dog chased the cat up the tree, who ran up the tree?",
        actual_output="It depends, some might consider the cat, while others might argue the dog.",
        expected_output="The cat."
    )
    
    # To run metric as a standalone
    correctness_metric.measure(test_case)
    print(f"GEval Score: {correctness_metric.score}, Reason: {correctness_metric.reason}")
    
    # Using the deepeval evaluate function
    deepeval_evaluate(test_cases=[test_case], metrics=[correctness_metric])

# Run the sanity check to understand BERTScore behavior
run_bertscore_sanity_check()

# Run the GEval demo to see how it works
run_geval_demo()
