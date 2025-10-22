from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import csv
import torch
from tqdm import tqdm
from datasets import load_dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run BLIP VQA inference on a dataset")
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="molvision/Clintox-V-SMILES-0",
        help="Name of the dataset on Hugging Face"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset split to use"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=str,
        default = "molvision/BLIP2-Clintox-Finetuned",
        help="Path to the fine-tuned model (local path or Hugging Face model ID)"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Salesforce/blip-vqa-base",
        help="Base model name for processor"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="neurips",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="VLM_inference_results.csv",
        help="Name of the output CSV file"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length for generated answers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_file_path = os.path.join(args.output_dir, args.output_file)

    print(f"Loading dataset: {args.dataset_name} (split: {args.dataset_split})...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    print(f"Loading processor from: {args.base_model_name}...")
    processor = BlipProcessor.from_pretrained(args.base_model_name)

    print(f"Loading model from: {args.model_path}...")
    model = BlipForQuestionAnswering.from_pretrained(args.model_path)
    
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        use_fp16 = True
        print("Using CUDA with FP16")
    else:
        model = model.to("cpu")
        use_fp16 = False
        print("Using CPU")
    
    model.eval()
    print(f"Model loaded successfully from {args.model_path}")
    
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Index", "Question", "Prediction", "Ground_Truth", 
            "Target_Molecule", "Sample_Method", "Sample_Num"
        ])
        
        print(f"\nProcessing {len(dataset)} samples...")
        
        # Process each sample
        for idx, sample in enumerate(tqdm(dataset, desc="Inference")):
            try:
                question = sample['Question']
                ground_truth = sample['Answer']
                target_molecule = sample.get('TargetMolecule', 'N/A')
                sample_method = sample.get('SampleMethod', 'N/A')
                sample_num = sample.get('SampleNum', 'N/A')
                image = sample['image']
                
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                with torch.no_grad():
                    encoding = processor(image, question, return_tensors="pt")

                    if use_fp16:
                        encoding = encoding.to(args.device, torch.float16)
                    else:
                        encoding = encoding.to(args.device)
                    
                    # Generate prediction
                    out = model.generate(**encoding, max_length=args.max_length)
                    prediction = processor.decode(out[0], skip_special_tokens=True)
                
                # Write result immediately
                csv_writer.writerow([
                    idx, question, prediction, ground_truth, 
                    target_molecule, sample_method, sample_num
                ])
                csv_file.flush()  # Ensure data is written to disk
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                csv_writer.writerow([
                    idx, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"
                ])
                csv_file.flush()
                continue
    
    print(f"\nInference completed! Results saved to {csv_file_path}")

if __name__ == "__main__":
    main()