import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
CHINESE_MODELS = {
    'bert': 'hfl/chinese-bert-wwm-ext',
    'roberta': 'hfl/chinese-roberta-wwm-ext',
    'albert': 'uw-madison/mbert-base-chinese',
    'macbert': 'hfl/chinese-macbert-base'
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 6,
    'max_length': 256,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'seed': 42
}

# Data split ratios
DATA_SPLIT = {
    'train': 0.8,
    'test': 0.2
}

# Label mapping
LABEL_MAP = {
    'stereotype': 1,
    'non-stereotype': 0,
    'offensive': 1,
    'non-offensive': 0
}

# COLD dataset specific settings
COLD_CONFIG = {
    'url': 'https://github.com/thu-coai/COLDataset',
    'bias_types': ['gender', 'region', 'race', 'occupation'],
    'target_bias_types': ['gender', 'region']  
}

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Results directory: {RESULTS_DIR}")

