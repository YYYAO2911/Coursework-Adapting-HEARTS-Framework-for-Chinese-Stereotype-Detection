import os
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from codecarbon import EmissionsTracker

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    CHINESE_MODELS, TRAINING_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable progress bar
os.environ["HUGGINGFACE_TRAINER_ENABLE_PROGRESS_BAR"] = "1"

def get_long_path(path_str):
    abs_path = os.path.abspath(path_str)
    if os.name == 'nt' and len(abs_path) > 200:
        if not abs_path.startswith('\\\\?\\'):
            return '\\\\?\\' + abs_path
    return abs_path

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Ensure required columns exist
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add group column if missing
    if 'group' not in df.columns:
        df['group'] = 'general'
    
    # Add data_name column if missing
    if 'data_name' not in df.columns:
        df['data_name'] = 'COLD'
    
    return df

def train_model(train_data, model_path, model_name, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_labels = len(train_data['label'].unique())
    logger.info(f"Number of unique labels: {num_labels}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f" GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA version: {torch.version.cuda}")
    else:
        logger.warning(" No GPU detected! Training will be slower on CPU.")
    
    # Start emissions tracking
    tracker = EmissionsTracker()
    tracker.start()
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding=True, 
            truncation=True, 
            max_length=TRAINING_CONFIG['max_length']
        )
    
    # Split train data for validation
    train_df, val_df = train_test_split(
        train_data, 
        test_size=0.2, 
        random_state=seed,
        stratify=train_data['label']
    )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    train_dataset = train_dataset.map(lambda x: {'labels': x['label']})
    
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(lambda x: {'labels': x['label']})
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "balanced_accuracy": balanced_acc
        }
    
    # Setup output directory
    model_output_dir = MODELS_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = str(model_output_dir)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=TRAINING_CONFIG['num_epochs'],
        eval_strategy="epoch",
        learning_rate=TRAINING_CONFIG['learning_rate'],
        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        use_cpu=False if torch.cuda.is_available() else True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(model_output_dir)
    logger.info(f"Model saved to: {model_output_dir}")
    
    # Stop emissions tracking
    emissions = tracker.stop()
    logger.info(f"Estimated total emissions: {emissions} kg CO2")
    
    return model_output_dir

def evaluate_model(test_data, model_output_dir, result_name, seed=42):

    np.random.seed(seed)
    
    num_labels = len(test_data['label'].unique())
    logger.info(f"Number of unique labels: {num_labels}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding=True, 
            truncation=True, 
            max_length=TRAINING_CONFIG['max_length']
        )
    
    # Create test dataset
    test_dataset = Dataset.from_pandas(test_data).map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(lambda x: {'labels': x['label']})
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"Evaluating on device: {device}")
    
    # Get predictions
    from transformers import pipeline
    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device_id
    )
    
    predictions = pipe(test_data['text'].tolist(), return_all_scores=True)
    pred_labels = [int(max(pred, key=lambda x: x['score'])['label'].split('_')[-1]) for pred in predictions]
    pred_probs = [max(pred, key=lambda x: x['score'])['score'] for pred in predictions]
    y_true = test_data['label'].tolist()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'data_name': test_data['data_name']
    })
    
    # Setup results directory
    result_output_dir = RESULTS_DIR / result_name
    result_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = get_long_path(str(result_output_dir / "full_results.csv"))
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved: {results_file}")
    
    # Generate and save classification report
    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    report_file = get_long_path(str(result_output_dir / "classification_report.csv"))
    df_report.to_csv(report_file, encoding='utf-8-sig')
    logger.info(f"Saved: {report_file}")
    
    # Print report
    print("\n" + "="*60)
    print(f"Classification Report - {result_name}")
    print("="*60)
    print(classification_report(y_true, pred_labels))
    
    return df_report

def main():

    print("="*60)
    print("Chinese Stereotype Detection - Training Pipeline")
    print("="*60)
    
    # Load data
    train_path = PROCESSED_DATA_DIR / "cold_train.csv"
    test_path = PROCESSED_DATA_DIR / "cold_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("\nError: Processed data not found!")
        print("Please run the following scripts first:")
        print("  1. python download_cold_dataset.py")
        print("  2. python data_preprocessing.py")
        return
    
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    print(f"\nLoaded data:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Train Chinese BERT model
    print("\n" + "="*60)
    print("Training Chinese BERT (chinese-bert-wwm-ext)")
    print("="*60)
    
    model_dir = train_model(
        train_data=train_data,
        model_path=CHINESE_MODELS['bert'],
        model_name='chinese_bert_cold',
        seed=TRAINING_CONFIG['seed']
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set")
    print("="*60)
    
    report = evaluate_model(
        test_data=test_data,
        model_output_dir=model_dir,
        result_name='chinese_bert_cold',
        seed=TRAINING_CONFIG['seed']
    )
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)
    print(f"\nModel saved to: {model_dir}")
    print(f"Results saved to: {RESULTS_DIR / 'chinese_bert_cold'}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary - Macro F1 Score")
    print("="*60)
    macro_f1 = report.loc['macro avg', 'f1-score']
    print(f"Chinese BERT on COLD dataset: {macro_f1:.4f} ({macro_f1*100:.2f}%)")

if __name__ == "__main__":
    main()

