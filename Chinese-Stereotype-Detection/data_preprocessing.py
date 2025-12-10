import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_SPLIT, TRAINING_CONFIG

def load_cold_data(filepath):

    df = pd.read_csv(filepath)
    return df

def clean_chinese_text(text):

    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (both Chinese and English style)
    text = re.sub(r'#\S+#?', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Chinese, English, numbers, and basic punctuation
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？、；：""''（）]', '', text)
    
    return text.strip()

def process_cold_dataset(include_bias_types=None):

    print("=" * 60)
    print("Processing COLD Dataset")
    print("=" * 60)
    
    # Load all data files (CSV format)
    all_dfs = []
    
    for filename in ['train.csv', 'dev.csv', 'test.csv']:
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            data = load_cold_data(filepath)
            print(f"Loaded {len(data)} samples from {filename}")
            all_dfs.append(data)
        else:
            print(f"Warning: {filename} not found")
    
    if not all_dfs:
        raise FileNotFoundError("No COLD data files found. Please run download_cold_dataset.py first.")
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples loaded: {len(df)}")
    
    # Display column info
    print(f"\nDataset columns: {list(df.columns)}")
    
    # Check for label column (COLD uses 'label')
    label_col = None
    for col in ['label', 'Label', 'offensive', 'Offensive']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Cannot find label column in dataset")
    
    # Check for text column (COLD uses 'TEXT')
    text_col = None
    for col in ['TEXT', 'text', 'content', 'sentence', 'Text', 'Content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Cannot find text column in dataset")
    
    print(f"\nUsing columns: text='{text_col}', label='{label_col}'")
    
    # Standardize column names
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    
    # Clean text
    print("\nCleaning text...")
    df['text'] = df['text'].apply(clean_chinese_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    print(f"Samples after cleaning: {len(df)}")
    
    # Convert labels to binary (0: non-stereotype/non-offensive, 1: stereotype/offensive)
    print("\nLabel distribution before processing:")
    print(df['label'].value_counts())
    
    # Handle different label formats
    if df['label'].dtype == 'object':
        # String labels
        label_map = {
            'offensive': 1, 'non-offensive': 0,
            'stereotype': 1, 'non-stereotype': 0,
            'yes': 1, 'no': 0,
            '1': 1, '0': 0
        }
        df['label'] = df['label'].str.lower().map(label_map)
    else:
        # Numeric labels - ensure binary
        df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
    
    # Remove any rows with NaN labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("\nLabel distribution after processing:")
    print(df['label'].value_counts())
    
    # Add dataset name column
    df['data_name'] = 'COLD'
    
    # If bias type column exists, filter by requested types (COLD uses 'topic')
    bias_col = None
    for col in ['topic', 'bias_type', 'type', 'category']:
        if col in df.columns:
            bias_col = col
            break
    
    if bias_col:
        print(f"\nTopic/bias types in dataset:")
        print(df[bias_col].value_counts())
        
        if include_bias_types:
            print(f"\nFiltering by bias types: {include_bias_types}")
            df = df[df[bias_col].isin(include_bias_types)]
            print(f"Samples after filtering: {len(df)}")
        
        df['group'] = df[bias_col]
    else:
        df['group'] = 'general'
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df, 
        test_size=DATA_SPLIT['test'], 
        random_state=TRAINING_CONFIG['seed'],
        stratify=df['label']
    )
    
    print(f"\nFinal dataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Display sample data
    print("\nSample training data:")
    print(train_df[['text', 'label', 'group']].head())
    
    return train_df, test_df

def save_processed_data(train_df, test_df, prefix='cold'):
    """
    Save processed data to CSV files
    """
    train_path = PROCESSED_DATA_DIR / f"{prefix}_train.csv"
    test_path = PROCESSED_DATA_DIR / f"{prefix}_test.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print(f"\nSaved processed data:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    return train_path, test_path

def analyze_dataset(df, name="Dataset"):
    """
    Provide detailed analysis of the dataset
    """
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {name}")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique texts: {df['text'].nunique()}")
    
    print(f"\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        label_name = 'Stereotype/Offensive' if label == 1 else 'Non-stereotype/Non-offensive'
        print(f"  {label_name} ({label}): {count} ({pct:.1f}%)")
    
    print(f"\nText Length Statistics:")
    text_lengths = df['text'].str.len()
    print(f"  Min length: {text_lengths.min()}")
    print(f"  Max length: {text_lengths.max()}")
    print(f"  Mean length: {text_lengths.mean():.1f}")
    print(f"  Median length: {text_lengths.median()}")
    
    if 'group' in df.columns:
        print(f"\nGroup Distribution:")
        group_counts = df['group'].value_counts()
        for group, count in group_counts.head(10).items():
            print(f"  {group}: {count}")

if __name__ == "__main__":
    # Process COLD dataset
    try:
        train_df, test_df = process_cold_dataset()
        
        # Save processed data
        save_processed_data(train_df, test_df)
        
        # Analyze datasets
        analyze_dataset(train_df, "Training Set")
        analyze_dataset(test_df, "Test Set")
        
        print("\n" + "="*60)
        print("Data preprocessing complete!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run download_cold_dataset.py first to download the data.")

