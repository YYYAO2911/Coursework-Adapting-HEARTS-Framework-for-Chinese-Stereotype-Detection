# Quick fix script: Only run the missing evaluation
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Get absolute path of current directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Data loader function
def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples):
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])
    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)
    combined_data['data_name'] = dataset_name
    
    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(combined_data, train_size=sample_proportion, stratify=combined_data['label'], random_state=42)
    
    train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42, stratify=sampled_data['label'])
    return train_data, test_data

def merge_datasets(train_data_candidate, test_data_candidate, train_data_established, test_data_established):
    merged_train_data = pd.concat([train_data_candidate, train_data_established], ignore_index=True)
    merged_test_data = pd.concat([test_data_candidate, test_data_established], ignore_index=True)
    return merged_train_data, merged_test_data

def evaluate_model(test_data, model_output_dir, result_output_base_dir, dataset_name, seed=42):
    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")
    
    # Use absolute path for model
    model_abs_path = str(SCRIPT_DIR / model_output_dir)
    print(f"Model path (absolute): {model_abs_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_abs_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_abs_path)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
    
    tokenized_test = Dataset.from_pandas(test_data).map(tokenize_function, batched=True).map(
        lambda examples: {'labels': examples['label']})
    
    # Create output directory using absolute path
    result_output_dir = SCRIPT_DIR / result_output_base_dir / dataset_name
    result_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (absolute): {result_output_dir}")
    
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    predictions = pipe(test_data['text'].to_list(), return_all_scores=True)
    pred_labels = [int(max(pred, key=lambda x: x['score'])['label'].split('_')[-1]) for pred in predictions]
    pred_probs = [max(pred, key=lambda x: x['score'])['score'] for pred in predictions]
    y_true = test_data['label'].tolist()
    
    # Save full results
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })
    results_file_path = str(result_output_dir / "full_results.csv")
    results_df.to_csv(results_file_path, index=False)
    print(f"Saved: {results_file_path}")
    
    # Save classification report
    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    result_file_path = str(result_output_dir / "classification_report.csv")
    
    # Debug: check if file exists and path is valid
    print(f"Attempting to save to: {result_file_path}")
    print(f"Directory exists: {result_output_dir.exists()}")
    print(f"File path type: {type(result_file_path)}")
    
    # Use explicit file writing to avoid pandas issues
    csv_content = df_report.to_csv()
    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"Saved: {result_file_path}")
    
    return df_report

# Load data (same as main script) - use absolute paths
print("Loading data...")
print(f"Script directory: {SCRIPT_DIR}")
train_data_winoqueer, test_data_winoqueer = data_loader(str(SCRIPT_DIR / 'Winoqueer - GPT Augmentation.csv'), 'stereotype', 'Winoqueer - GPT Augmentation', 1000000, 5)
train_data_seegull, test_data_seegull = data_loader(str(SCRIPT_DIR / 'SeeGULL - GPT Augmentation.csv'), 'stereotype', 'SeeGULL - GPT Augmentation', 1000000, 5)
train_data_mgsd, test_data_mgsd = data_loader(str(SCRIPT_DIR / 'MGSD.csv'), 'stereotype', 'MGSD', 1000000, 5)

train_merged_winoqueer, test_merged_winoqueer = merge_datasets(train_data_winoqueer, test_data_winoqueer, train_data_mgsd, test_data_mgsd)
train_merged_seegull, test_merged_seegull = merge_datasets(train_data_seegull, test_data_seegull, train_data_mgsd, test_data_mgsd)
train_merged_all, test_merged_all = merge_datasets(train_data_seegull, test_data_seegull, train_merged_winoqueer, test_merged_winoqueer)

print(f"\nTest data merged size: {len(test_merged_all)}")

# Run ONLY the missing evaluation - SAVE TO SHORTER PATH FIRST
print("\n" + "="*60)
print("Running missing evaluation")
print("="*60 + "\n")

from sklearn.metrics import classification_report as sk_report
import torch

# Load model
model_path = str(SCRIPT_DIR / 'model_output_albertv2' / 'merged_winoqueer_seegull_gpt_augmentation_trained')
print(f"Loading model from: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Get predictions
predictions = pipe(test_merged_all['text'].to_list(), return_all_scores=True)
pred_labels = [int(max(pred, key=lambda x: x['score'])['label'].split('_')[-1]) for pred in predictions]
y_true = test_merged_all['label'].tolist()

# Generate report
report = sk_report(y_true, pred_labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# SAVE TO CURRENT DIRECTORY FIRST (short path), then copy
temp_file = str(SCRIPT_DIR / "temp_report.csv")
print(f"Saving to temp file: {temp_file}")
df_report.to_csv(temp_file)

# Now copy to target location using Windows xcopy (handles long paths better)
target_dir = r"result_output_albertv2\merged_winoqueer_seegull_gpt_augmentation_trained\merged_winoqueer_seegull_gpt_augmentation"
target_file = os.path.join(str(SCRIPT_DIR), target_dir, "classification_report.csv")

# Read and write using Python
with open(temp_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Try to write using Windows long path prefix
long_path = "\\\\?\\" + target_file
print(f"Writing to long path: {long_path}")
with open(long_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Clean up temp file
os.remove(temp_file)

print(f"\n✅ Done! Classification report saved.")
print(f"Report preview:\n{df_report}")

print("\n✅ Done! Missing evaluation completed.")

