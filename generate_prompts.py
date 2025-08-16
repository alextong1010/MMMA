import argparse
import json
import os
import yaml
from datasets import load_dataset
from utils import *
from tqdm import tqdm

def main(config_path):
    config = load_config(config_path)
    local_dataset_dir = config['local_dataset_dir'] 
    dataset_name = config['dataset_name']
    split = config['split']
    model_name = config['model']
    dataset_config = load_dataset_config(dataset_name, split)
    dataset_path = dataset_config['path']
    
    # Load dataset
    print("Loading dataset...")
    current_dir = os.getcwd()
    dataset = load_dataset(dataset_path)

    local_dataset_path = os.path.join(local_dataset_dir, dataset_name)

    # Pick split
    split_ds = dataset[split]
    len_split_ds = len(split_ds)

    # Load model client for token counting
    client = load_client(model_name)

    # Create output directory following the same pattern as other scripts
    output_dir = os.path.join(current_dir, dataset_name, split, "generated_prompts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    output_file = os.path.join(output_dir, f"prompts_{dataset_name}_{split}.json")

    # Build prompt dict
    if dataset_name == "MathVista":
        prompt_data = {}
        for row in tqdm(split_ds, total=len_split_ds, desc="Generating prompts"):
            pid = row["pid"]
            image_path = row["image"]  # Get the image path from the dataset
            image_path = os.path.join(local_dataset_path, image_path)
            prompt = MathVista_make_prompt(row)
            # Count tokens for the prompt (This takes a while)
            token_count = count_tokens(client, model_name, prompt)
            prompt_data[pid] = [image_path, prompt, token_count]  # Store as [image_path, prompt, token_count]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")

    # Save
    save_json(prompt_data, output_file)
    print(f"Saved {len(prompt_data)} prompts for split '{split}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate prompts from dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args.config) 
