import pandas as pd
import argparse
import re
from sklearn.metrics import accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model predictions from CSV')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path containing predictions'
    )
    parser.add_argument(
        '--ground_truth_column',
        type=str,
        default='Answer',
        help='Column name containing ground truth labels (default: Answer)'
    )
    parser.add_argument(
        '--prediction_column',
        type=str,
        default='predictions',
        help='Column name containing model predictions (default: predictions)'
    )
    return parser.parse_args()

def extract_boolean_answer(text):
    """
    Extract boolean answer from text.
    Handles: <boolean>Yes</boolean> or plain Yes/No in text
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text)
    
    # Try to extract from <boolean> tags
    boolean_pattern = r'<boolean>\s*(Yes|No|YES|NO|yes|no)\s*</boolean>'
    match = re.search(boolean_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Look for Yes/No in text
    patterns = [
        r'\b(Yes|No|YES|NO|yes|no)\b',
        r'^(Yes|No|YES|NO|yes|no)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def normalize_label(label):
    """Normalize to Yes/No for comparison"""
    if pd.isna(label) or label is None:
        return None
    label_str = str(label).strip()
    if label_str.lower() == 'yes':
        return 'Yes'
    elif label_str.lower() == 'no':
        return 'No'
    return None

def main():
    args = parse_args()
    
    # Load CSV
    df = pd.read_csv(args.input)
    
    # Extract predictions
    df['extracted_prediction'] = df[args.prediction_column].apply(extract_boolean_answer)
    
    # Normalize labels
    df['normalized_ground_truth'] = df[args.ground_truth_column].apply(normalize_label)
    df['normalized_prediction'] = df['extracted_prediction'].apply(normalize_label)
    
    # Filter valid predictions
    valid_df = df[df['normalized_prediction'].notna() & df['normalized_ground_truth'].notna()]
    
    if len(valid_df) == 0:
        print("ERROR: No valid predictions found!")
        return
    
    y_true = valid_df['normalized_ground_truth'].values
    y_pred = valid_df['normalized_prediction'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='Yes')
    
    # Print results
    print(f"\nTotal Samples: {len(df)}")
    print(f"Valid Predictions: {len(valid_df)}")
    print(f"Failed Extractions: {len(df) - len(valid_df)}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()