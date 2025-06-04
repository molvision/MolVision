# main.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from models.gpt import GPTInferencer

def main(args):
    # Create a directory for downloaded files
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
    os.makedirs(download_dir, exist_ok=True)
    
    # First, list all files in the repository to understand the structure
    try:
        print(f"Listing files in repository: {args.dataset}")
        all_files = list_repo_files(repo_id=args.dataset, repo_type="dataset")
        
        # Filter for metadata and image files
        metadata_files = [f for f in all_files if f.endswith('.csv') or f.endswith('.jsonl')]
        image_files = [f for f in all_files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        
        print(f"Found {len(metadata_files)} metadata files and {len(image_files)} image files")
        
        # Find the most likely metadata file
        metadata_file_path = None
        for pattern in [f"{args.split}/metadata.csv", "metadata.csv", f"{args.split}.csv"]:
            if pattern in metadata_files:
                metadata_file_path = pattern
                break
        
        if not metadata_file_path and metadata_files:
            # Just use the first metadata file
            metadata_file_path = metadata_files[0]
        
        if not metadata_file_path:
            raise ValueError("No metadata file found in repository")
            
        print(f"Using metadata file: {metadata_file_path}")
        
        # Download the metadata file
        metadata_file = hf_hub_download(
            repo_id=args.dataset,
            filename=metadata_file_path,
            repo_type="dataset",
            local_dir=download_dir
        )
        
        # Read the metadata
        if metadata_file_path.endswith('.csv'):
            dataset = pd.read_csv(metadata_file)
        else:  # jsonl
            import json
            data = []
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset = pd.DataFrame(data)
            
        print(f"Successfully loaded {len(dataset)} samples from metadata")
        
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return
    
    # Limit the number of samples if specified
    if args.num_samples > 0:
        dataset = dataset.head(args.num_samples)
        print(f"\nLimited to {len(dataset)} samples")
    
    # Initialize GPT inferencer
    gpt = GPTInferencer()
    
    # Prepare results list
    results = []
    
    # Create a directory for downloaded images
    image_dir = os.path.join(download_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
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
            
            # Handle image - ALWAYS use forward slashes for HF Hub paths
            image_file = f"train/images/molecule_{idx}.png"
            
            # Download the image
            try:
                print(f"Downloading image: {image_file}")
                image_path = hf_hub_download(
                    repo_id=args.dataset,
                    filename=image_file,
                    repo_type="dataset",
                    local_dir=image_dir
                )
                print(f"Downloaded image to: {image_path}")
                image = image_path
                
            except Exception as img_e:
                print(f"Error downloading image {image_file}: {img_e}")
                print("Continuing with text-only inference")
                image = None
            
            # Run inference
            result = gpt.infer(question, image=image, model=args.model)
            
            # Store results
            result_entry = {
                'sample_id': idx,
                'question': question,
                'image_path': image if image else None,
                'has_image': image is not None,
                'gpt_response': result
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
    parser = argparse.ArgumentParser(description='Run GPT inference on HuggingFace dataset')
    
    parser.add_argument('--dataset', type=str, default='ChemVision/BACE-V-SMILES-2',
                        help='HuggingFace dataset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='GPT model to use (gpt-4o or gpt-4v)')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file path')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process (default: 10, -1 for all)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save results after every N samples (0 to disable)')
                        
    args = parser.parse_args()
    main(args)