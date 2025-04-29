# RAG System Evaluation Summary
Date: 2025-04-14 16:38:00

## Retrieval Performance
- Precision: 0.3220                  <!-- Gives percentage of retrieved documents that are relevant  -->
- Recall: 0.4000                     <!-- Gives percentage of relevant documents that are successfully retrieved  -->
- MRR (Mean Reciprocal Rank): 0.3900 <!-- Measures how high the first relevant documents appears in your results  -->
- Hit Rate: 0.4200                   <!-- Gives % of queries where at least one relevant documents was retrieved  -->

## Generation Performance
### OPENAI -turbo model
- BERTScore Precision: 0.8590        <!-- Precision - how much of the generated content is in the reference   -->
- BERTScore Recall: 0.8835           <!-- Recall - how much of the reference content is in the generation -->
- BERTScore F1: 0.8708               <!-- F1 - Harmonic mean of precision and recall -->
- Semantic Similarity: 0.7978        <!-- Cosine similarity between embeddings of generated and reference answers -->
- Answer Relevance: 0.7400           <!-- How relevant the generated answer is to the question (LLM-as-a-judge) -->
- Factual Accuracy: 0.7200           <!-- How factually correct the generated answer is (LLM-as-a-judge) -->
- Groundedness: 0.6600               <!-- How well the generated answer is grounded in the retrieved documents -->

### LLAMA -phi model
- BERTScore Precision: 0.8044
- BERTScore Recall: 0.8645
- BERTScore F1: 0.8329
- Semantic Similarity: 0.7104
- Answer Relevance: 0.7100
- Factual Accuracy: 0.7000
- Groundedness: 0.6400