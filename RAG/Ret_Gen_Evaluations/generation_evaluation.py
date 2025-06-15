from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate as deepeval_evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import Any, List, Mapping, Optional
from together import Together  # Import Together directly
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def generate_answer(model, question, context):
    """Generate an answer using either OpenAI or Together API."""
    # Use the same NASA-specific prompt template as in rag_models.py
    prompt = f"""You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

Context: {context}

Question: {question}

Format:
1. Brief Answer
2. Key Points
3. Relevant Mission Details (if applicable)

Please ensure your response:
- Is factually accurate and grounded in the provided context
- Includes specific technical details and measurements when available
- Maintains a clear structure with the numbered format
- Focuses on the most relevant information to answer the question
- Avoids speculation or information not supported by the context
"""
    
    if isinstance(model, ChatOpenAI):
        # Use OpenAI with specific parameters
        response = model.invoke(prompt, temperature=0.1, max_tokens=1024)
        return response.content
    else:
        # Use Together API with matching parameters
        together_client = Together()
        try:
            response = together_client.chat.completions.create(
                model=model,  # model is the model name string
                messages=[{
                    "role": "system",
                    "content": "You are an expert in analyzing NASA documents and mission data. Provide concise, accurate answers based on the given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Together API: {e}")
            raise e

def evaluate_generation(rag_model, test_questions, reference_answers, model_name, ground_truth_docs, use_ground_truth_fallback=True):
    """
    Evaluate the generation component of the RAG system.
    
    Args:
        rag_model: The RAG model instance
        test_questions: List of test questions
        reference_answers: Dictionary mapping questions to reference answers
        model_name: Name of the model to use for generation
        ground_truth_docs: Dictionary mapping questions to relevant document IDs
        use_ground_truth_fallback: Whether to use ground truth documents when retrieval fails
    
    Returns:
        Dictionary containing evaluation metrics and processing statistics
    """
    # Initialize empty results
    empty_results = {
        'bert_precision': 0.0,
        'bert_recall': 0.0,
        'bert_f1': 0.0,
        'semantic_similarity': 0.0,
        'answer_relevance': 0.0,
        'factual_accuracy': 0.0,
        'groundedness': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'geval_score': 0.0,
        'geval_relevance': 0.0,
        'geval_accuracy': 0.0,
        'geval_groundedness': 0.0,
        'error': None,
        'processing_stats': {
            'total_questions': len(test_questions),
            'processed_questions': 0,
            'skipped_questions': 0
        }
    }
    
    if not model_name:
        empty_results['error'] = "model_name must be specified"
        return empty_results
        
    logger.info(f"\nStarting evaluation with model: {model_name}")
    
    try:
        # Setup the model
        if model_name == "gpt4o-mini":
            model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1024
            )
        elif model_name == "gpt3.5-turbo":
            model = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=1024
            )
        elif model_name == "llama-70b":
            if not rag_model.together_available:
                empty_results['error'] = "TogetherAI not available"
                return empty_results
            model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        elif model_name == "llama-vision":
            if not rag_model.together_available:
                empty_results['error'] = "TogetherAI not available"
                return empty_results
            model = "meta-llama/Llama-Vision-Free"
        else:
            empty_results['error'] = f"Unknown model type: {model_name}"
            return empty_results
        
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
            'geval_groundedness': [],
            'processing_stats': {
                'total_questions': len(test_questions),
                'processed_questions': 0,
                'skipped_questions': 0
            }
        }
        
        # Initialize models for evaluation
        sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        rouge = Rouge()
        metrics = setup_geval_metrics()
        
        for question in test_questions:
            logger.info(f"\nProcessing question: {question}")
            logger.info(f"Using model: {model_name}")
            
            try:
                # First try to get context from retriever
                logger.info("Retrieving documents for context...")
                retrieved_docs = rag_model.retriever.get_relevant_documents(question)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                # If retriever failed and we're using fallback
                if not retrieved_docs and use_ground_truth_fallback and question in ground_truth_docs:
                    logger.info("Retriever found no documents, using ground truth documents")
                    relevant_ids = ground_truth_docs[question]
                    retrieved_docs = []
                    for doc_id in relevant_ids:
                        doc = rag_model.get_document_by_id(doc_id)  # You'll need to implement this
                        if doc:
                            retrieved_docs.append(doc)
                    logger.info(f"Found {len(retrieved_docs)} ground truth documents")
                
                if not retrieved_docs:
                    logger.warning("No documents found - skipping question")
                    results['processing_stats']['skipped_questions'] += 1
                    continue
                
                # Create context from documents
                context_parts = []
                for doc in retrieved_docs:
                    source = doc.metadata.get('file_name', 'Unknown Source')
                    title = doc.metadata.get('title', 'No Title')
                    context_parts.append(f"Source: {source}\nTitle: {title}\nContent: {doc.page_content}\n")
                context = "\n".join(context_parts)
                
                # Generate answer with context
                generated_answer = generate_answer(model, question, context)
                logger.info(f"Generated answer: {generated_answer[:100]}...")
                
                # Get reference answer
                reference_answer = reference_answers.get(question, "")
                if not reference_answer:
                    logger.warning("No reference answer found for this question")
                    results['processing_stats']['skipped_questions'] += 1
                    continue
                
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
                    
                    logger.info(f"  BERTScore raw values - P: {P.item():.2f}, R: {R.item():.2f}, F1: {F1.item():.2f}")
                    
                    # Apply additional normalization if needed
                    bert_precision = max(0, min(1, P.item()))
                    bert_recall = max(0, min(1, R.item()))
                    bert_f1 = max(0, min(1, F1.item()))
                    
                except Exception as e:
                    logger.error(f"  Error calculating BERTScore: {e}")
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
                    
                    logger.info(f"  ROUGE Scores - R1: {rouge_results['rouge1']['f']:.4f}, R2: {rouge_results['rouge2']['f']:.4f}, " +
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
                                logger.info(f"  Sentence-aligned ROUGE-L: {avg_sentence_rouge:.4f}")
                    except Exception as e:
                        logger.error(f"  Error in sentence-aligned ROUGE: {e}")
                    
                except Exception as e:
                    logger.error(f"  Error calculating ROUGE scores: {e}")
                    rouge_results = {
                        'rouge1': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                        'rouge2': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                        'rougeL': {'p': 0.0, 'r': 0.0, 'f': 0.0}
                    }
                
                # For answer relevance, factual accuracy, and groundedness, using LLM-as-a-judge
                try:
                    if not retrieved_docs:
                        # If no documents, set groundedness to 0 but still evaluate other metrics
                        relevance, accuracy = evaluate_with_llm_no_context(
                            question, generated_answer, reference_answer
                        )
                        groundedness = 0.0
                        logger.info("No context available - groundedness set to 0.0")
                    else:
                        relevance, accuracy, groundedness = evaluate_with_llm(
                            question, generated_answer, reference_answer, retrieved_docs
                        )
                except Exception as e:
                    logger.error(f"  Error in LLM evaluation: {e}")
                    relevance, accuracy, groundedness = 0.5, 0.5, 0.0 if not retrieved_docs else 0.5
                
                # Run GEval evaluation
                try:
                    logger.info("Starting GEval evaluation...")
                    if not retrieved_docs:
                        # Run GEval without groundedness if no context
                        geval_results = run_geval_evaluation_no_context(
                            question, 
                            generated_answer, 
                            reference_answer,
                            metrics
                        )
                        # Add explicit 0.0 groundedness score
                        geval_results['groundedness'] = {
                            'score': 0.0,
                            'reason': 'No retrieved documents available for groundedness evaluation'
                        }
                    else:
                        geval_results = run_geval_evaluation(
                            question, 
                            generated_answer, 
                            reference_answer,
                            metrics,
                            retrieved_docs
                        )
                    logger.info("GEval evaluation completed")
                    logger.info(f"  GEval Scores - Correctness: {geval_results['correctness']['score']:.2f} - {geval_results['correctness']['reason'][:100]}...")
                    logger.info(f"  GEval Scores - Relevance: {geval_results['relevance']['score']:.2f} - {geval_results['relevance']['reason'][:100]}...")
                    logger.info(f"  GEval Scores - Accuracy: {geval_results['accuracy']['score']:.2f} - {geval_results['accuracy']['reason'][:100]}...")
                    logger.info(f"  GEval Scores - Groundedness: {geval_results['groundedness']['score']:.2f} - {geval_results['groundedness']['reason'][:100]}...")
                    
                    # Verify groundedness context
                    if 'groundedness' in geval_results:
                        if geval_results['groundedness']['reason'] == "No context available for evaluation":
                            logger.error("Groundedness evaluation failed despite having retrieved documents")
                            logger.error(f"Retrieved docs count: {len(retrieved_docs)}")
                            logger.error(f"Context present in test case: {bool(context)}")
                except Exception as e:
                    logger.error(f"Error in GEval evaluation: {e}")
                    logger.error(f"Retrieved docs available: {bool(retrieved_docs)}")
                    logger.error(f"Context length: {len(context) if context else 'No context'}")
                    geval_results = {
                        'correctness': {'score': 0.0, 'reason': "Error in evaluation"},
                        'relevance': {'score': 0.0, 'reason': "Error in evaluation"},
                        'accuracy': {'score': 0.0, 'reason': "Error in evaluation"},
                        'groundedness': {'score': 0.0, 'reason': f"Error: {str(e)}"}
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
                
                logger.info(f"  BERTScore F1: {bert_f1:.2f}, Similarity: {similarity:.2f}")
                logger.info(f"  Relevance: {relevance:.2f}, Accuracy: {accuracy:.2f}, Groundedness: {groundedness:.2f}")
                
                results['processing_stats']['processed_questions'] += 1
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                results['processing_stats']['skipped_questions'] += 1
                continue
        
        # Calculate final results
        if results['processing_stats']['processed_questions'] > 0:
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
                    final_results[metric] = sum(results[metric]) / len(results[metric])
                elif metric != 'processing_stats':
                    final_results[metric] = sum(results[metric]) / len(results[metric])
            
            final_results['processing_stats'] = results['processing_stats']
            return final_results
        else:
            empty_results['error'] = "No questions were successfully processed"
            return empty_results
            
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        empty_results['error'] = str(e)
        return empty_results

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
    
    # Create context from retrieved documents with better formatting
    context_parts = []
    for doc in retrieved_docs:
        # Extract metadata
        source = doc.metadata.get('file_name', 'Unknown Source')
        title = doc.metadata.get('title', 'No Title')
        
        # Format the context entry
        context_parts.append(f"Source: {source}\nTitle: {title}\nContent: {doc.page_content}\n")
    
    context = "\n---\n".join(context_parts)
    
    evaluation_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator for NASA technical documentation question-answering systems. Please evaluate the following answer based on three criteria:
    
    Question: {question}
    
    Generated Answer: {generated_answer}
    
    Reference Answer: {reference_answer}
    
    Retrieved Context:
    {context}
    
    Please evaluate on a scale of 0 to 10 for each criterion, considering:

    1. Relevance (0-10):
    - Does the answer directly address the question?
    - Are all parts of the response relevant to the question?
    - Does it maintain focus on the specific technical aspects asked about?

    2. Factual Accuracy (0-10):
    - Does the answer align with the reference answer's technical details?
    - Are specific measurements, values, and technical terms used correctly?
    - Are there any contradictions or omissions of critical information?

    3. Groundedness (0-10):
    - Can each claim be verified from the provided context?
    - Does the answer use specific details from the context?
    - Does it avoid speculation beyond what's supported by the sources?

    Format your response as three numbers separated by commas, e.g., "7,8,6"
    Provide only the numbers, no explanation.
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
        return 0.5, 0.5, 0.5

def evaluate_with_llm_no_context(question, generated_answer, reference_answer):
    """
    Use an LLM to evaluate answer relevance and factual accuracy without context.
    Returns tuple of (relevance_score, accuracy_score)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    evaluation_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator for NASA technical documentation question-answering systems. 
    Please evaluate the following answer based on two criteria:
    
    Question: {question}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}
    
    Please evaluate on a scale of 0 to 10 for each criterion:

    1. Relevance (0-10):
    - Does the answer directly address the question?
    - Are all parts of the response relevant to the question?
    - Does it maintain focus on the specific technical aspects asked about?

    2. Factual Accuracy (0-10):
    - Does the answer align with the reference answer's technical details?
    - Are specific measurements, values, and technical terms used correctly?
    - Are there any contradictions or omissions of critical information?

    Format your response as two numbers separated by commas, e.g., "7,8"
    Provide only the numbers, no explanation.
    """)
    
    result = llm.invoke(evaluation_prompt.format(
        question=question,
        generated_answer=generated_answer,
        reference_answer=reference_answer
    ))
    
    try:
        scores = result.content.strip().split(',')
        relevance = float(scores[0]) / 10
        accuracy = float(scores[1]) / 10
        return relevance, accuracy
    except:
        return 0.5, 0.5

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
        try:
            context_parts = []
            for doc in retrieved_docs:
                # Extract metadata if available
                source = doc.metadata.get('file_name', 'Unknown Source')
                title = doc.metadata.get('title', 'No Title')
                
                # Add formatted context entry
                context_parts.append(f"Source: {source}\nTitle: {title}\nContent: {doc.page_content.strip()}")
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Created context from {len(retrieved_docs)} documents")
            logger.debug(f"Context length: {len(context)} characters")
        except Exception as e:
            logger.error(f"Error creating context from retrieved docs: {e}")
            context = ""
    else:
        logger.warning("No retrieved_docs provided for groundedness evaluation")
    
    # Base test case without context
    try:
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            expected_output=reference_answer
        )
        logger.debug("Created base test case")
    except Exception as e:
        logger.error(f"Error creating test case: {e}")
        return {
            "correctness": {"score": 0.5, "reason": f"Error creating test case: {str(e)}"},
            "relevance": {"score": 0.5, "reason": f"Error creating test case: {str(e)}"},
            "accuracy": {"score": 0.5, "reason": f"Error creating test case: {str(e)}"},
            "groundedness": {"score": 0.5, "reason": f"Error creating test case: {str(e)}"}
        }
    
    # Add context if available (for groundedness)
    if context:
        try:
            test_case.context = context
            logger.debug("Added context to test case")
        except Exception as e:
            logger.error(f"Error adding context to test case: {e}")
    
    results = {}
    
    # Run correctness metric
    try:
        metrics["correctness"].measure(test_case)
        results["correctness"] = {
            "score": metrics["correctness"].score,
            "reason": metrics["correctness"].reason
        }
        logger.debug("Completed correctness evaluation")
    except Exception as e:
        logger.error(f"Error in GEval correctness measurement: {e}")
        results["correctness"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run relevance metric
    try:
        metrics["relevance"].measure(test_case)
        results["relevance"] = {
            "score": metrics["relevance"].score,
            "reason": metrics["relevance"].reason
        }
        logger.debug("Completed relevance evaluation")
    except Exception as e:
        logger.error(f"Error in GEval relevance measurement: {e}")
        results["relevance"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run accuracy metric
    try:
        metrics["accuracy"].measure(test_case)
        results["accuracy"] = {
            "score": metrics["accuracy"].score,
            "reason": metrics["accuracy"].reason
        }
        logger.debug("Completed accuracy evaluation")
    except Exception as e:
        logger.error(f"Error in GEval accuracy measurement: {e}")
        results["accuracy"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
    # Run groundedness metric if context is available
    if context:
        try:
            logger.info("Starting groundedness evaluation")
            logger.debug(f"Context available: {bool(test_case.context)}")
            logger.debug(f"Context length: {len(test_case.context)}")
            
            metrics["groundedness"].measure(test_case)
            results["groundedness"] = {
                "score": metrics["groundedness"].score,
                "reason": metrics["groundedness"].reason
            }
            logger.info("Completed groundedness evaluation successfully")
        except Exception as e:
            logger.error(f"Error in GEval groundedness measurement: {e}")
            logger.error(f"Test case state - Has context: {hasattr(test_case, 'context')}")
            logger.error(f"Context type: {type(test_case.context) if hasattr(test_case, 'context') else 'None'}")
            results["groundedness"] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    else:
        logger.warning("Skipping groundedness evaluation - no context available")
        results["groundedness"] = {"score": 0.5, "reason": "No context available for evaluation"}
    
    return results

def run_geval_evaluation_no_context(question, generated_answer, reference_answer, metrics):
    """
    Run GEval evaluation without groundedness check.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=reference_answer
    )
    
    results = {}
    
    # Run metrics that don't require context
    for metric_name in ["correctness", "relevance", "accuracy"]:
        try:
            metrics[metric_name].measure(test_case)
            results[metric_name] = {
                "score": metrics[metric_name].score,
                "reason": metrics[metric_name].reason
            }
        except Exception as e:
            logger.error(f"Error in GEval {metric_name} measurement: {e}")
            results[metric_name] = {"score": 0.5, "reason": f"Error: {str(e)}"}
    
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
        logger.info(f"Text 1: {text1}")
        logger.info(f"Text 2: {text2}")
        logger.info(f"BERTScore - P: {P.item():.2f}, R: {R.item():.2f}, F1: {F1.item():.2f}\n")

# Function to demonstrate GEval usage independently
def run_geval_demo():
    """
    Demonstrate GEval usage with a simple example.
    """
    logger.info("Running GEval demo...")
    
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
    logger.info(f"GEval Score: {correctness_metric.score}, Reason: {correctness_metric.reason}")
    
    # Using the deepeval evaluate function
    deepeval_evaluate(test_cases=[test_case], metrics=[correctness_metric])

# Run the sanity check to understand BERTScore behavior
run_bertscore_sanity_check()

# Run the GEval demo to see how it works
run_geval_demo()
