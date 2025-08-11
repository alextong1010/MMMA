import os
import argparse
import time
from tqdm import tqdm
from utils import *

def main(config_path):
    """
    This script is used to generate solutions from saved prompts
    """
    # Load config
    config = load_config(config_path)
    dataset_name = config['dataset_name']
    split = config['split']
    model_name = config['model']

    current_dir = os.getcwd()

    # Load model
    client = load_client(model_name)

    # Load dataset from saved prompts
    print("Loading dataset...")
    prompt_dir = os.path.join(current_dir, dataset_name, split, "generated_prompts")
    prompts = load_json(os.path.join(prompt_dir, f"prompts_{dataset_name}_{split}.json"))
    len_prompts = len(prompts)
    print(f"Loaded {len_prompts} prompts")

    print("Generating solutions...")
    # Create output directory
    output_dir = os.path.join(current_dir, dataset_name, split, f"generated_solutions_{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"solutions_{dataset_name}_{split}.json")

    # Generate solutions, prompts is a dict with pid as key and [image_path, prompt] as value
    solutions = {}
    for pid, prompt_data in tqdm(prompts.items(), total=len_prompts, desc="Generating solutions"):
        image_path = prompt_data[0]
        image = client.files.upload(file=image_path)
        prompt = prompt_data[1]

        # Note: I put the image in a list because the generate_solution function expects a list of images (for potential future use of multiple images)
        solution = generate_solution(client, model_name, [image], prompt)
        solutions[pid] = solution

        # add a delay to avoid rate limit
        # time.sleep(1) # delay of 2 seconds to avoid rate limit of 30 RPM
        # Note make sure it doesnt go above 14,400 requests per day for Gemma 3 API calls
    
    # Save all solutions once as a single JSON object {pid: solution}
    save_json(solutions, output_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample random images from dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args.config) 
