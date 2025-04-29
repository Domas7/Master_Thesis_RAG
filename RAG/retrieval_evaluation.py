import numpy as np

def evaluate_retrieval(rag_model, test_questions, ground_truth_docs):
    """
    Evaluate the retrieval component of the RAG system.
    
    Args:
        rag_model: Your RAG model instance
        test_questions: List of test questions
        ground_truth_docs: Dictionary mapping questions to relevant document IDs
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        'precision': [],
        'recall': [],
        'mrr': [],
        'hit_rate': [],
        'ndcg': []
    }
    
    for question in test_questions:
        print(f"\nEvaluating retrieval for question: {question}")
        
        # Get retrieved documents
        # Note: We need to temporarily modify the retriever to return more documents for evaluation
        original_k = rag_model.retriever.search_kwargs["k"]
        rag_model.retriever.search_kwargs["k"] = 10  # Retrieve more docs for evaluation
        
        retrieved_docs = rag_model.retriever.get_relevant_documents(question)
        retrieved_ids = []
        for doc in retrieved_docs:
            if doc.metadata.get('source_type') == 'lessons_learned':
                retrieved_ids.append(doc.metadata.get('url', ''))
            else:
                retrieved_ids.append(doc.metadata.get('chunk_id', ''))
        
        # Restore original k value
        rag_model.retriever.search_kwargs["k"] = original_k
        
        # Get ground truth for this question
        relevant_ids = ground_truth_docs.get(question, [])
        
        # DEBUG: Print expected vs. retrieved IDs
        print(f"  Expected document IDs: {relevant_ids}")
        print(f"  Retrieved document IDs: {retrieved_ids}")
        
        # Check for partial matches (in case of formatting differences)
        if len(set(retrieved_ids) & set(relevant_ids)) == 0:
            print("  No exact matches found. Checking for partial matches...")
            partial_matches = []
            for rel_id in relevant_ids:
                for ret_id in retrieved_ids:
                    # Check if one is a substring of the other
                    if rel_id in ret_id or ret_id in rel_id:
                        print(f"  Partial match found: '{rel_id}' ~ '{ret_id}'")
                        partial_matches.append((rel_id, ret_id))
            
            if partial_matches:
                print("  Consider updating your ground truth IDs to match the retriever's format")
        
        # Calculate metrics
        precision = len(set(retrieved_ids) & set(relevant_ids)) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0
        
        # MRR calculation
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        # Hit rate
        hit_rate = 1 if len(set(retrieved_ids) & set(relevant_ids)) > 0 else 0
        
        # NDCG calculation
        ndcg = calculate_ndcg(retrieved_ids, relevant_ids)
        
        # Store results
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['mrr'].append(mrr)
        results['hit_rate'].append(hit_rate)
        results['ndcg'].append(ndcg)
        
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, MRR: {mrr:.2f}, NDCG: {ndcg:.2f}")
        
        print("  Retrieved documents details:")
        for i, doc in enumerate(retrieved_docs):
            source_type = doc.metadata.get('source_type', 'unknown')
            doc_id = doc.metadata.get('chunk_id', doc.metadata.get('url', 'unknown'))
            title = doc.metadata.get('title', 'unknown')
            print(f"    {i+1}. Type: {source_type}, ID: {doc_id}, Title: {title}")
            # Print a short snippet of content
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"       Content: {content_preview}")
    
    # Average the results
    final_results = {}
    for metric in results:
        final_results[metric] = sum(results[metric]) / len(results[metric]) if results[metric] else 0.0
    
    # Print summary statistics
    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Total questions evaluated: {len(test_questions)}")
    print(f"Questions with at least one relevant document retrieved: {sum(results['hit_rate'])}")
    print(f"Average precision: {final_results['precision']:.4f}")
    print(f"Average recall: {final_results['recall']:.4f}")
    print(f"Average MRR: {final_results['mrr']:.4f}")
    
    return final_results

def calculate_ndcg(retrieved_ids, relevant_ids, k=None):
    """Calculate NDCG for a single query."""
    if k is None:
        k = len(retrieved_ids)
    
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    dcg = 0
    idcg = 0
    
    # Calculate DCG
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1 if doc_id in relevant_ids else 0
        dcg += rel / np.log2(i + 2)
    
    # Calculate IDCG
    for i in range(min(len(relevant_ids), k)):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0
