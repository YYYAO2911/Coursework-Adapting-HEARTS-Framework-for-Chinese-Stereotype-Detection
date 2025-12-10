## Table: Test Set Macro F1 Score - Original Paper vs Reproduction Results

| Model Type | Training Data | Test Set | Original | Reproduced | Difference |
|------------|---------------|----------|----------|------------|------------|
| **DistilRoBERTa-Bias** | wikirev-bias | MGSD | 53.1% | 53.1% | 0.0% |
| **DistilRoBERTa-Bias** | wikirev-bias | AWinoQueer | 59.7% | 59.7% | 0.0% |
| **DistilRoBERTa-Bias** | wikirev-bias | ASeeGULL | 65.5% | 65.5% | 0.0% |
| **DistilRoBERTa-Bias** | wikirev-bias | EMGSD | 53.9% | 53.9% | 0.0% |
| | | | | | |
| **LR - TFIDF** | MGSD | MGSD | 65.7% | 65.7% | 0.0% |
| **LR - TFIDF** | MGSD | AWinoQueer | 53.2% | 53.2% | 0.0% |
| **LR - TFIDF** | MGSD | ASeeGULL | 67.3% | 67.3% | 0.0% |
| **LR - TFIDF** | MGSD | EMGSD | 65.0% | 65.0% | 0.0% |
| **LR - TFIDF** | AWinoQueer | MGSD | 49.8% | 49.8% | 0.0% |
| **LR - TFIDF** | AWinoQueer | AWinoQueer | 95.6% | 95.6% | 0.0% |
| **LR - TFIDF** | AWinoQueer | ASeeGULL | 59.7% | 59.7% | 0.0% |
| **LR - TFIDF** | AWinoQueer | EMGSD | 52.7% | 52.7% | 0.0% |
| **LR - TFIDF** | ASeeGULL | MGSD | 57.4% | 57.4% | 0.0% |
| **LR - TFIDF** | ASeeGULL | AWinoQueer | 56.7% | 56.7% | 0.0% |
| **LR - TFIDF** | ASeeGULL | ASeeGULL | 82.0% | 82.0% | 0.0% |
| **LR - TFIDF** | ASeeGULL | EMGSD | 58.3% | 58.3% | 0.0% |
| **LR - TFIDF** | EMGSD | MGSD | 65.8% | 65.8% | 0.0% |
| **LR - TFIDF** | EMGSD | AWinoQueer | 83.1% | 83.1% | 0.0% |
| **LR - TFIDF** | EMGSD | ASeeGULL | 76.2% | 76.2% | 0.0% |
| **LR - TFIDF** | EMGSD | EMGSD | 67.2% | 67.2% | 0.0% |
| | | | | | |
| **LR - Embeddings** | MGSD | MGSD | 61.6% | 62.6% | +1.0% |
| **LR - Embeddings** | MGSD | AWinoQueer | 63.3% | 64.7% | +1.4% |
| **LR - Embeddings** | MGSD | ASeeGULL | 71.7% | 67.5% | -4.2% |
| **LR - Embeddings** | MGSD | EMGSD | 62.1% | 63.0% | +0.9% |
| **LR - Embeddings** | AWinoQueer | MGSD | 55.5% | 58.4% | +2.9% |
| **LR - Embeddings** | AWinoQueer | AWinoQueer | 93.9% | 93.2% | -0.7% |
| **LR - Embeddings** | AWinoQueer | ASeeGULL | 66.1% | 68.7% | +2.6% |
| **LR - Embeddings** | AWinoQueer | EMGSD | 58.4% | 60.8% | +2.4% |
| **LR - Embeddings** | ASeeGULL | MGSD | 53.5% | 59.7% | **+6.2%** |
| **LR - Embeddings** | ASeeGULL | AWinoQueer | 56.8% | 79.2% | **+22.4%** |
| **LR - Embeddings** | ASeeGULL | ASeeGULL | 86.0% | 86.9% | +0.9% |
| **LR - Embeddings** | ASeeGULL | EMGSD | 54.9% | 62.1% | **+7.2%** |
| **LR - Embeddings** | EMGSD | MGSD | 62.1% | 62.7% | +0.6% |
| **LR - Embeddings** | EMGSD | AWinoQueer | 75.4% | 77.5% | +2.1% |
| **LR - Embeddings** | EMGSD | ASeeGULL | 76.7% | 74.2% | -2.5% |
| **LR - Embeddings** | EMGSD | EMGSD | 63.4% | 64.0% | +0.6% |
| | | | | | |
| **ALBERT-V2** | MGSD | MGSD | 79.7% | 79.0% | -0.7% |
| **ALBERT-V2** | MGSD | AWinoQueer | 74.7% | 72.5% | -2.2% |
| **ALBERT-V2** | MGSD | ASeeGULL | 75.9% | 73.8% | -2.1% |
| **ALBERT-V2** | MGSD | EMGSD | 79.3% | 78.5% | -0.8% |
| **ALBERT-V2** | AWinoQueer | MGSD | 60.0% | 61.0% | +1.0% |
| **ALBERT-V2** | AWinoQueer | AWinoQueer | 97.3% | 98.6% | +1.3% |
| **ALBERT-V2** | AWinoQueer | ASeeGULL | 70.7% | 71.3% | +0.6% |
| **ALBERT-V2** | AWinoQueer | EMGSD | 62.8% | 63.8% | +1.0% |
| **ALBERT-V2** | ASeeGULL | MGSD | 63.1% | 63.2% | +0.1% |
| **ALBERT-V2** | ASeeGULL | AWinoQueer | 66.8% | 47.6% | **-19.2%** |
| **ALBERT-V2** | ASeeGULL | ASeeGULL | 88.4% | 86.8% | -1.6% |
| **ALBERT-V2** | ASeeGULL | EMGSD | 64.5% | 63.4% | -1.1% |
| **ALBERT-V2** | EMGSD | MGSD | 80.2% | 78.5% | -1.7% |
| **ALBERT-V2** | EMGSD | AWinoQueer | 97.4% | 97.6% | +0.2% |
| **ALBERT-V2** | EMGSD | ASeeGULL | 87.3% | 87.9% | +0.6% |
| **ALBERT-V2** | EMGSD | EMGSD | 81.5% | 79.9% | -1.6% |
| | | | | | |
| **DistilBERT** | MGSD | MGSD | 78.3% | 78.6% | +0.3% |
| **DistilBERT** | MGSD | AWinoQueer | 75.6% | 76.0% | +0.4% |
| **DistilBERT** | MGSD | ASeeGULL | 73.0% | 74.0% | +1.0% |
| **DistilBERT** | MGSD | EMGSD | 78.0% | 78.3% | +0.3% |
| **DistilBERT** | AWinoQueer | MGSD | 61.1% | 60.9% | -0.2% |
| **DistilBERT** | AWinoQueer | AWinoQueer | 98.1% | 98.5% | +0.4% |
| **DistilBERT** | AWinoQueer | ASeeGULL | 72.1% | 74.7% | +2.6% |
| **DistilBERT** | AWinoQueer | EMGSD | 64.0% | 64.0% | 0.0% |
| **DistilBERT** | ASeeGULL | MGSD | 62.7% | 62.8% | +0.1% |
| **DistilBERT** | ASeeGULL | AWinoQueer | 82.1% | 86.0% | +3.9% |
| **DistilBERT** | ASeeGULL | ASeeGULL | 89.8% | 88.9% | -0.9% |
| **DistilBERT** | ASeeGULL | EMGSD | 65.1% | 65.3% | +0.2% |
| **DistilBERT** | EMGSD | MGSD | 79.0% | 79.1% | +0.1% |
| **DistilBERT** | EMGSD | AWinoQueer | 98.8% | 98.6% | -0.2% |
| **DistilBERT** | EMGSD | ASeeGULL | 91.9% | 90.9% | -1.0% |
| **DistilBERT** | EMGSD | EMGSD | 80.6% | 80.6% | 0.0% |
| | | | | | |
| **BERT** | MGSD | MGSD | 81.2% | 81.0% | -0.2% |
| **BERT** | MGSD | AWinoQueer | 77.9% | 75.7% | -2.2% |
| **BERT** | MGSD | ASeeGULL | 69.9% | 74.2% | +4.3% |
| **BERT** | MGSD | EMGSD | 80.6% | 80.5% | -0.1% |
| **BERT** | AWinoQueer | MGSD | 59.1% | 62.7% | +3.6% |
| **BERT** | AWinoQueer | AWinoQueer | 97.9% | 98.5% | +0.6% |
| **BERT** | AWinoQueer | ASeeGULL | 72.5% | 71.9% | -0.6% |
| **BERT** | AWinoQueer | EMGSD | 62.3% | 65.1% | +2.8% |
| **BERT** | ASeeGULL | MGSD | 61.0% | 63.7% | +2.7% |
| **BERT** | ASeeGULL | AWinoQueer | 78.6% | 81.1% | +2.5% |
| **BERT** | ASeeGULL | ASeeGULL | 89.6% | 88.5% | -1.1% |
| **BERT** | ASeeGULL | EMGSD | 63.3% | 65.7% | +2.4% |
| **BERT** | EMGSD | MGSD | 81.7% | 81.3% | -0.4% |
| **BERT** | EMGSD | AWinoQueer | 97.6% | 97.4% | -0.2% |
| **BERT** | EMGSD | ASeeGULL | 88.9% | 90.4% | +1.5% |
| **BERT** | EMGSD | EMGSD | 82.8% | 82.5% | -0.3% |

---

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Experiments | 84 |
| Within ±5% | 79 (94.0%) |
| Outside ±5% | 5 (6.0%) |
| Average Absolute Difference | 1.8% |
| Max Positive Difference | +22.4% (LR-Embed ASeeGULL→AWinoQueer) |
| Max Negative Difference | -19.2% (ALBERT-V2 ASeeGULL→AWinoQueer) |

*Note: Differences marked in **bold** exceed the ±5% threshold.*