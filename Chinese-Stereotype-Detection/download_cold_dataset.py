"""
Download and prepare COLD (Chinese Offensive Language Dataset) dataset
Source: https://github.com/thu-coai/COLDataset

"""
import os
import requests
import json
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def download_cold_dataset():
    
    # Correct URLs for COLD dataset (CSV format in COLDataset subfolder)
    base_url = "https://raw.githubusercontent.com/thu-coai/COLDataset/main/COLDataset"
    
    files_to_download = {
        'train.csv': f"{base_url}/train.csv",
        'dev.csv': f"{base_url}/dev.csv",
        'test.csv': f"{base_url}/test.csv"
    }
    
    print("=" * 60)
    print("Downloading COLD (Chinese Offensive Language Dataset)")
    print("=" * 60)
    
    success_count = 0
    
    for filename, url in files_to_download.items():
        output_path = RAW_DATA_DIR / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            success_count += 1
            continue
            
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✓ Saved to {output_path}")
            success_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  Please manually download from: {url}")
    
    if success_count == len(files_to_download):
        print("\n✓ All files downloaded successfully!")
    else:
        print(f"\n  Downloaded {success_count}/{len(files_to_download)} files")
        print("\nManual download steps:")
        print("1. Visit: https://github.com/thu-coai/COLDataset/tree/main/COLDataset")
        print("2. Download train.csv, dev.csv, test.csv")
        print(f"3. Put the files in: {RAW_DATA_DIR}")
    
    print("\n" + "=" * 60)
    print("Download completed!")
    print("=" * 60)

def inspect_cold_dataset():
    """
    Inspect the downloaded COLD dataset structure
    """
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("Inspecting COLD Dataset")
    print("=" * 60)
    
    for filename in ['train.csv', 'dev.csv', 'test.csv']:
        filepath = RAW_DATA_DIR / filename
        
        if not filepath.exists():
            print(f"✗ {filename} not found")
            continue
        
        try:
            df = pd.read_csv(filepath)
            
            print(f"\n{filename}:")
            print(f"  - Number of samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            
            # Show label distribution if label column exists
            for col in ['label', 'Label', 'offensive']:
                if col in df.columns:
                    print(f"  - Label distribution:")
                    for label, count in df[col].value_counts().items():
                        print(f"      {label}: {count}")
                    break
            
            # Show sample
            print(f"  - Sample row:")
            sample = df.iloc[0].to_dict()
            for key, value in list(sample.items())[:4]:
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"      {key}: {value}")
                
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")

def create_manual_download_instructions():
    """
    Create instructions for manual download if automatic download fails
    """
    instructions = f"""
# COLD Dataset Manual Download Instructions

If automatic download fails, please manually download the dataset:

1. Go to: https://github.com/thu-coai/COLDataset

2. Download the following files from the 'data' folder:
   - train.json
   - dev.json  
   - test.json

3. Place them in: {RAW_DATA_DIR}

## Alternative: Clone the repository

git clone https://github.com/thu-coai/COLDataset.git

Then copy the files from COLDataset/data/ to your project's data/raw/ folder.

## Dataset Description

COLD (Chinese Offensive Language Dataset) contains:
- 37,480 annotated comments from Chinese social media
- Categories: offensive vs non-offensive
- Topics: race, gender, region, occupation, etc.
- Source: Tsinghua University CoAI Group

## Citation

If you use this dataset, please cite:

@inproceedings{{deng2022cold,
  title={{COLD: A Benchmark for Chinese Offensive Language Detection}},
  author={{Deng, Jiawen and Zhou, Jingyan and Sun, Hao and others}},
  booktitle={{Proceedings of EMNLP}},
  year={{2022}}
}}
"""
    
    readme_path = RAW_DATA_DIR / "DOWNLOAD_INSTRUCTIONS.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\nDownload instructions saved to: {readme_path}")

if __name__ == "__main__":
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    download_cold_dataset()
    
    # Create manual download instructions
    create_manual_download_instructions()
    
    # Inspect dataset
    inspect_cold_dataset()

