# main.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from models.model_config import ModelConfig

def main(args):
    # Create a directory for downloaded files
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
    os.makedirs(download_dir, exist_ok=True)
    
    # Load dataset from HuggingFace
    try:
        print(f"Loading dataset: {args.dataset}, split: {args.split}")
        hf_dataset = load_dataset(args.dataset, split=args.split)
        dataset = hf_dataset.to_pandas()
        print(f"Successfully loaded {len(dataset)} samples from dataset")
        print(f"Columns: {list(dataset.columns)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Limit the number of samples if specified
    if args.num_samples > 0:
        dataset = dataset.head(args.num_samples)
        print(f"\nLimited to {len(dataset)} samples")
    
    # Initialize model inferencer using ModelConfig
    print(f"Initializing {args.model} model...")
    inferencer, model_param = ModelConfig.get_model_inferencer(args.model)
    print(f"Model initialized successfully")
    
    # Prepare results list
    results = []
    
    # Process each sample in the dataset
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing samples"):
        try:
            # Extract the question
            question = None
            for field in ['Question', 'question', 'prompt', 'text', 'input']:
                if field in dataset.columns and not pd.isna(row[field]):
                    question = row[field]
                    break
            
            if not question:
                print(f"Warning: Sample {idx} has no usable question/prompt field")
                question = "No prompt available for this sample"
            
            # Handle image if present in dataset
            image = None
            for field in ['image', 'Image', 'images']:
                if field in dataset.columns and not pd.isna(row[field]):
                    image = row[field]
                    break
            
            # Run inference
            if model_param:
                result = inferencer.infer(question, image=image, model=model_param)
            else:
                result = inferencer.infer(question, image=image)
            
            # Store results
            result_entry = {
                'sample_id': idx,
                'question': question,
                'has_image': image is not None,
                'model': args.model,
                'response': result
            }
            
            results.append(result_entry)
            
            # Save intermediate results
            if args.save_interval > 0 and (len(results) % args.save_interval == 0):
                pd.DataFrame(results).to_csv(args.output, index=False)
                
        except Exception as sample_e:
            print(f"Error processing sample {idx}: {sample_e}")
            continue
            
    # Save final results
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model inference on HuggingFace dataset')
    
    parser.add_argument('--dataset', type=str, default='molvision/BACE-V-SMILES-4',
                        help='HuggingFace dataset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        choices=ModelConfig.get_available_models(),
                        help='Model to use for inference')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file path')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process (default: 10, -1 for all)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save results after every N samples (0 to disable)')
                        
    args = parser.parse_args()
    main(args)