import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, PROCESSED_DATA_DIR

# Original HEARTS paper baseline results (Macro F1 scores)
# EMGSD = MGSD + WinoQueer + SeeGULL (57,201 samples total)
# Results below are from models trained on EMGSD, tested on EMGSD

ORIGINAL_RESULTS_EMGSD = {
    # Trained on EMGSD, tested on EMGSD (from Table 2 of HEARTS paper)
    'BERT': 0.828,           # 82.8%
    'ALBERT-V2': 0.815,      # 81.5% (actually 79.9% in reproduction)
    'DistilBERT': 0.806,     # 80.6%
    'LR-TFIDF': 0.672,       # 67.2%
    'LR-Embeddings': 0.634,  # 63.4%
    'DistilRoBERTa-Bias': 0.539,  # 53.9% (pre-trained on different data)
}

# For reference: Results when trained on MGSD only, tested on MGSD
ORIGINAL_RESULTS_MGSD = {
    'BERT': 0.812,           # 81.2%
    'ALBERT-V2': 0.797,      # 79.7%
    'DistilBERT': 0.783,     # 78.3%
    'LR-TFIDF': 0.657,       # 65.7%
    'LR-Embeddings': 0.616,  # 61.6%
    'DistilRoBERTa-Bias': 0.531,  # 53.1%
}

# Use EMGSD results as primary comparison 
ORIGINAL_RESULTS = ORIGINAL_RESULTS_EMGSD

def load_results(result_dir):

    report_path = result_dir / "classification_report.csv"
    if not report_path.exists():
        return None
    return pd.read_csv(report_path, index_col=0)

def load_full_results(result_dir):

    results_path = result_dir / "full_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)

def compare_with_original(adapted_f1, model_name="Chinese BERT"):

    print("\n" + "="*70)
    print("Comparison with Original HEARTS Paper Results")
    print("="*70)
    
    print(f"\nAdapted Model ({model_name}): {adapted_f1:.4f} ({adapted_f1*100:.2f}%)")
    print("\nOriginal Paper Baselines (Macro F1 on EMGSD):")
    print("Note: EMGSD = MGSD + WinoQueer + SeeGULL (57,201 English samples)")
    print("-"*50)
    
    comparison_data = []
    
    for model, original_f1 in ORIGINAL_RESULTS.items():
        diff = adapted_f1 - original_f1
        diff_pct = (diff / original_f1) * 100
        
        status = "✓" if abs(diff_pct) <= 5 else "✗"
        
        print(f"  {model}: {original_f1:.3f} | Δ = {diff:+.3f} ({diff_pct:+.1f}%) {status}")
        
        comparison_data.append({
            'Original Model': model,
            'Original F1': original_f1,
            'Adapted F1': adapted_f1,
            'Difference': diff,
            'Difference (%)': diff_pct,
            'Within ±5%': abs(diff_pct) <= 5
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "-"*50)
    avg_original = np.mean(list(ORIGINAL_RESULTS.values()))
    diff_avg = adapted_f1 - avg_original
    print(f"Average Original: {avg_original:.3f}")
    print(f"Difference from Average: {diff_avg:+.3f} ({(diff_avg/avg_original)*100:+.1f}%)")
    
    return comparison_df

def statistical_significance_test(y_true, y_pred_1, y_pred_2, name1="Model 1", name2="Model 2"):

    print("\n" + "="*70)
    print(f"Statistical Significance Test: {name1} vs {name2}")
    print("="*70)
    
    # Create contingency table
    # a: both correct, b: model1 correct & model2 wrong
    # c: model1 wrong & model2 correct, d: both wrong
    
    correct_1 = (np.array(y_true) == np.array(y_pred_1))
    correct_2 = (np.array(y_true) == np.array(y_pred_2))
    
    a = np.sum(correct_1 & correct_2)  # Both correct
    b = np.sum(correct_1 & ~correct_2)  # Only model 1 correct
    c = np.sum(~correct_1 & correct_2)  # Only model 2 correct
    d = np.sum(~correct_1 & ~correct_2)  # Both wrong
    
    print(f"\nContingency Table:")
    print(f"                    {name2}")
    print(f"                    Correct  Wrong")
    print(f"{name1} Correct     {a:5d}   {b:5d}")
    print(f"{name1} Wrong       {c:5d}   {d:5d}")
    
    # McNemar's test
    if b + c > 0:
        # Chi-squared approximation
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        print(f"\nMcNemar's Test:")
        print(f"  Chi-squared statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("  Result: Statistically significant difference (p < 0.05)")
        else:
            print("  Result: No statistically significant difference (p >= 0.05)")
    else:
        print("\nCannot perform McNemar's test: b + c = 0")
        p_value = 1.0
    
    return p_value

def analyze_failure_cases(results_df, n_examples=10):

    print("\n" + "="*70)
    print("Failure Case Analysis")
    print("="*70)
    
    # Find misclassified samples
    failures = results_df[results_df['predicted_label'] != results_df['actual_label']]
    
    print(f"\nTotal samples: {len(results_df)}")
    print(f"Misclassified: {len(failures)} ({len(failures)/len(results_df)*100:.2f}%)")
    
    # Analyze by error type
    false_positives = failures[
        (failures['predicted_label'] == 1) & (failures['actual_label'] == 0)
    ]
    false_negatives = failures[
        (failures['predicted_label'] == 0) & (failures['actual_label'] == 1)
    ]
    
    print(f"\nError Breakdown:")
    print(f"  False Positives (predicted stereotype, actual non-stereotype): {len(false_positives)}")
    print(f"  False Negatives (predicted non-stereotype, actual stereotype): {len(false_negatives)}")
    
    # Show example failure cases
    if len(false_positives) > 0:
        print(f"\n--- False Positive Examples (Top {min(n_examples, len(false_positives))}) ---")
        for _, row in false_positives.head(n_examples).iterrows():
            text = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
            print(f"  Text: {text}")
            print(f"  Confidence: {row['predicted_probability']:.3f}")
            print()
    
    if len(false_negatives) > 0:
        print(f"\n--- False Negative Examples (Top {min(n_examples, len(false_negatives))}) ---")
        for _, row in false_negatives.head(n_examples).iterrows():
            text = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
            print(f"  Text: {text}")
            print(f"  Confidence: {row['predicted_probability']:.3f}")
            print()
    
    # Analyze by confidence
    print("\n--- Confidence Analysis ---")
    print(f"Average confidence (correct predictions): {results_df[results_df['predicted_label'] == results_df['actual_label']]['predicted_probability'].mean():.3f}")
    print(f"Average confidence (incorrect predictions): {failures['predicted_probability'].mean():.3f}")
    
    return failures

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Non-Stereotype', 'Stereotype'],
        yticklabels=['Non-Stereotype', 'Stereotype']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Chinese Stereotype Detection')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
    
    plt.close()

def generate_comparison_table():

    print("\n" + "="*70)
    print("Results Comparison Table")
    print("="*70)
    
    # Check for Chinese BERT results
    chinese_bert_dir = RESULTS_DIR / "chinese_bert_cold"
    
    if chinese_bert_dir.exists():
        report = load_results(chinese_bert_dir)
        if report is not None:
            adapted_f1 = report.loc['macro avg', 'f1-score']
            
            table = """
| Model | Training Data | Test Data | F1 Score | Notes |
|-------|---------------|-----------|----------|-------|
| **Original Models (HEARTS Paper - EMGSD)** |||||
| BERT | EMGSD | EMGSD | 82.8% | Fine-tuned |
| ALBERT-V2 | EMGSD | EMGSD | 81.5% | Fine-tuned |
| DistilBERT | EMGSD | EMGSD | 80.6% | Fine-tuned |
| LR-TFIDF | EMGSD | EMGSD | 67.2% | Logistic Regression |
| LR-Embeddings | EMGSD | EMGSD | 63.4% | spaCy embeddings |
| DistilRoBERTa-Bias | wikirev-bias | EMGSD | 53.9% | Pre-trained (diff data) |
| **Adapted Model (This Study)** |||||
| Chinese BERT | COLD | COLD | {:.1f}% | Adapted for Chinese |

Note: EMGSD (57,201 samples) = MGSD + WinoQueer + SeeGULL
      COLD (37,480 samples) = Chinese Offensive Language Dataset
""".format(adapted_f1 * 100)
            
            print(table)
            return table
    
    print("No results found. Please run train_chinese_bert.py first.")
    return None

def main():

    print("="*70)
    print("Chinese Stereotype Detection - Evaluation & Comparison")
    print("="*70)
    
    # Check for results
    chinese_bert_dir = RESULTS_DIR / "chinese_bert_cold"
    
    if not chinese_bert_dir.exists():
        print("\nNo results found!")
        print("Please run the training script first:")
        print("  python train_chinese_bert.py")
        return
    
    # Load results
    report = load_results(chinese_bert_dir)
    full_results = load_full_results(chinese_bert_dir)
    
    if report is None or full_results is None:
        print("Error loading results")
        return
    
    # Get metrics
    adapted_f1 = report.loc['macro avg', 'f1-score']
    
    print("\n" + "="*70)
    print("Chinese BERT Model Performance")
    print("="*70)
    print(report)
    
    # Compare with original
    comparison_df = compare_with_original(adapted_f1, "Chinese BERT")
    
    # Save comparison
    comparison_df.to_csv(RESULTS_DIR / "comparison_with_original.csv", index=False)
    
    # Analyze failure cases
    failures = analyze_failure_cases(full_results)
    
    # Save failure analysis
    failures.to_csv(RESULTS_DIR / "failure_cases.csv", index=False, encoding='utf-8-sig')
    
    # Generate comparison table
    generate_comparison_table()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        full_results['actual_label'],
        full_results['predicted_label'],
        save_path=RESULTS_DIR / "confusion_matrix.png"
    )
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()

