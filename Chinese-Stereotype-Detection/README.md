# Chinese Social Media Stereotype Detection

## 中文社交媒体刻板印象检测项目

### Project Overview

This project adapts the HEARTS (Holistic Framework for Assessing Responses of Text-based Stereotypes) approach to detect stereotypes in Chinese social media content, specifically focusing on **regional stereotypes** and **gender stereotypes** in Chinese context.

### SDG Alignment

- **SDG 10**: Reduced Inequalities - Addressing discrimination based on regional origin
- **SDG 5**: Gender Equality - Detecting gender-based stereotypes
- **SDG 16**: Peace, Justice and Strong Institutions - Promoting inclusive online discourse

### Dataset

Using the **COLD (Chinese Offensive Language Dataset)** from Tsinghua University:
- Source: https://github.com/thu-coai/COLDataset
- Contains labeled offensive/biased language from Chinese social media
- Includes various bias types including regional and gender stereotypes

### Project Structure

```
Chinese-Stereotype-Detection/
├── data/                    # Dataset storage
│   ├── raw/                 # Original COLD dataset
│   └── processed/           # Preprocessed data
├── models/                  # Trained model checkpoints
├── results/                 # Evaluation results
├── config.py                # Configuration settings
├── data_preprocessing.py    # Data preprocessing pipeline
├── train_chinese_bert.py    # Training script for Chinese BERT
├── evaluate.py              # Evaluation script
└── README.md
```

### Models Used

| Original (English) | Adapted (Chinese) |
|-------------------|-------------------|
| BERT | chinese-bert-wwm-ext |
| DistilBERT | chinese-roberta-wwm-ext |
| ALBERT | chinese-albert-base |

### Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full dependencies

### Usage

1. Download COLD dataset: `python download_cold_dataset.py`
2. Run data preprocessing: `python data_preprocessing.py`
3. Train Chinese BERT model: `python train_chinese_bert.py`
4. Evaluate and compare with original results: `python evaluate_compare.py`

### Results

| Model | Dataset | Macro F1 |
|-------|---------|----------|
| Chinese BERT | COLD | **90.46%** |

### Citations

If you use this project, please cite the following:

**COLD Dataset:**
```bibtex
@article{deng2022cold,
  title="Cold: A benchmark for chinese offensive language detection",
  author="Deng, Jiawen and Zhou, Jingyan and Sun, Hao and Mi, Fei and Huang, Minlie",
  booktitle="Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
  month=dec,
  year="2022",
  address="Abu Dhabi, United Arab Emirates",
  publisher="Association for Computational Linguistics",
  url="https://aclanthology.org/2022.emnlp-main.796",
  pages="11580--11599"
}
```

**HEARTS Framework (Original Paper):**
```bibtex
@article{king2024hearts,
  title="HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection",
  author="King, Theo and Wu, Zekun and Koshiyama, Adriano and Kazim, Emre and Treleaven, Philip",
  journal="arXiv preprint arXiv:2409.11579",
  year="2024"
}
```

**Chinese BERT:**
```bibtex
@article{cui2020revisiting,
  title="Revisiting Pre-Trained Models for Chinese Natural Language Processing",
  author="Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing",
  journal="Findings of EMNLP 2020",
  year="2020"
}
```


