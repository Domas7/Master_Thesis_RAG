# RAG System Evaluation Summary
Date: 2025-05-06 10:12:09

## Retrieval Performance
- Precision: 0.1520
- Recall: 0.2000
- MRR (Mean Reciprocal Rank): 0.2100
- Hit Rate: 0.2200
- NDCG: 0.1971

## Generation Performance
### OPENAI
#### Semantic Metrics
- BERTScore Precision: 0.5620
- BERTScore Recall: 0.5340
- BERTScore F1: 0.5474
- Semantic Similarity: 0.0327

#### ROUGE Metrics
- ROUGE-1 F1: 0.1059
- ROUGE-2 F1: 0.0058
- ROUGE-L F1: 0.1151

#### LLM-as-judge Metrics (custom implementation)
- Answer Relevance: 0.8600
- Factual Accuracy: 0.9000
- Groundedness: 0.7800

#### GEval Metrics (LLM-as-judge using deepeval)
- GEval Correctness: 0.0005
- GEval Relevance: 0.7620
- GEval Accuracy: 0.0000
- GEval Groundedness: 0.7779

#### Comparison between LLM evaluation methods
- Relevance difference: -0.0980 (-9.8%)
- Accuracy difference: -0.9000 (-90.0%)
- Groundedness difference: -0.0021 (-0.2%)

### LLAMA
#### Semantic Metrics
- BERTScore Precision: 0.5680
- BERTScore Recall: 0.5647
- BERTScore F1: 0.5659
- Semantic Similarity: 0.0648

#### ROUGE Metrics
- ROUGE-1 F1: 0.1281
- ROUGE-2 F1: 0.0164
- ROUGE-L F1: 0.1427

#### LLM-as-judge Metrics (custom implementation)
- Answer Relevance: 0.8000
- Factual Accuracy: 0.8400
- Groundedness: 0.7000

#### GEval Metrics (LLM-as-judge using deepeval)
- GEval Correctness: 0.0113
- GEval Relevance: 0.9481
- GEval Accuracy: 0.0000
- GEval Groundedness: 0.6151

#### Comparison between LLM evaluation methods
- Relevance difference: 0.1481 (+14.8%)
- Accuracy difference: -0.8400 (-84.0%)
- Groundedness difference: -0.0849 (-8.5%)
