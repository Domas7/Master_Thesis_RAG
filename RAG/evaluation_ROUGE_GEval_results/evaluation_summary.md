# RAG System Evaluation Summary
Date: 2025-05-06 09:53:10

## Retrieval Performance
- Precision: 0.5500
- Recall: 0.8000
- MRR (Mean Reciprocal Rank): 0.8000
- Hit Rate: 0.8000
- NDCG: 0.8000

## Generation Performance
### OPENAI
#### Semantic Metrics
- BERTScore Precision: 0.7476
- BERTScore Recall: 0.7983
- BERTScore F1: 0.7711
- Semantic Similarity: 0.7788

#### ROUGE Metrics
- ROUGE-1 F1: 0.4253
- ROUGE-2 F1: 0.2201
- ROUGE-L F1: 0.4026

#### LLM-as-judge Metrics (custom implementation)
- Answer Relevance: 0.7800
- Factual Accuracy: 0.8000
- Groundedness: 0.6400

#### GEval Metrics (LLM-as-judge using deepeval)
- GEval Correctness: 0.7135
- GEval Relevance: 0.7697
- GEval Accuracy: 0.7325
- GEval Groundedness: 0.7907

#### Comparison between LLM evaluation methods
- Relevance difference: -0.0103 (-1.0%)
- Accuracy difference: -0.0675 (-6.8%)
- Groundedness difference: 0.1507 (+15.1%)

### LLAMA
#### Semantic Metrics
- BERTScore Precision: 0.6804
- BERTScore Recall: 0.6891
- BERTScore F1: 0.6840
- Semantic Similarity: 0.6968

#### ROUGE Metrics
- ROUGE-1 F1: 0.2353
- ROUGE-2 F1: 0.0418
- ROUGE-L F1: 0.2240

#### LLM-as-judge Metrics (custom implementation)
- Answer Relevance: 0.6200
- Factual Accuracy: 0.5200
- Groundedness: 0.5200

#### GEval Metrics (LLM-as-judge using deepeval)
- GEval Correctness: 0.3202
- GEval Relevance: 0.6325
- GEval Accuracy: 0.3396
- GEval Groundedness: 0.4735

#### Comparison between LLM evaluation methods
- Relevance difference: 0.0125 (+1.2%)
- Accuracy difference: -0.1804 (-18.0%)
- Groundedness difference: -0.0465 (-4.6%)
