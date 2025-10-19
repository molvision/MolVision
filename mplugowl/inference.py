import torch
import argparse
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import pandas as pd
import base64
from datasets import load_dataset
import os

import io

def parse_args():
    parser = argparse.ArgumentParser(description='Run mPlug-Owl2 inference on a dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='molvision/BBBP-V-SMILES-2',
        help='HuggingFace dataset name (default: molvision/BBBP-V-SMILES-2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Results/mplugowl.csv',
        help='Output CSV file path (default: Results/mplugowl.csv)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='MAGAer13/mplug-owl2-llama2-7b',
        help='Model path (default: MAGAer13/mplug-owl2-llama2-7b)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum number of new tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1)'
    )
    return parser.parse_args()

def ensure_output_directory(output_path):
    """Ensure the output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

def main():
    args = parse_args()
    
    # Ensure output directory exists
    ensure_output_directory(args.output)
    
    # Load the dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, load_8bit=False, load_4bit=True, 
        device="cuda", device_map="auto"
    )
    
    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Add predictions column if it doesn't exist
    if 'predictions' not in df.columns:
        df['predictions'] = None
    
    # Check if output file exists to resume from last checkpoint
    start_index = 0
    if os.path.exists(args.output):
        existing_df = pd.read_csv(args.output)
        # Find the first row without a prediction
        for idx, pred in enumerate(existing_df['predictions']):
            if pd.isna(pred) or pred == '' or pred == 'None':
                start_index = idx
                break
        else:
            start_index = len(existing_df)
        print(f"Resuming from index {start_index}")
        df = existing_df
    
    print(f"Processing {len(df) - start_index} images...")
    
    # Process each row
    for index, row in df.iloc[start_index:].iterrows():
        print(f"\nProcessing image {index + 1}/{len(df)}")
        
        image = row['image']  # Directly get the image from the dataset
        query = row['Question']
        
        # Create new conversation for each image
        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles
        
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to a square
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        
        # Process the image
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        inp = DEFAULT_IMAGE_TOKEN + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        df.at[index, 'predictions'] = outputs
        
        # Ensure directory exists before saving (in case it was deleted during execution)
        ensure_output_directory(args.output)
        
        # Save after each inference (incremental update)
        df.to_csv(args.output, index=False)
        print(f"Saved prediction for row {index} to {args.output}")
    
    print(f"\nAll predictions completed and saved to {args.output}")

if __name__ == "__main__":
    main()