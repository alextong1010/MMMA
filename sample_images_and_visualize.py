import os
from datasets import load_dataset, Image
import argparse
import random
import numpy as np

from utils import *


def main(config_path):
    """Sample random images from dataset and save them"""

    config = load_config(config_path)
    dataset_dir = config['dataset_dir']
    dataset_name = config['dataset_name']
    split = config['split']
    n_samples = config['n_samples']
    seed = config['seed']

    # Get dataset config and config file
    dataset_config = load_dataset_config(dataset_name, split)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"🎲 Sampling {n_samples} random images with seed {seed}")
    print(f"Dataset: {dataset_name}")
    print("-" * 50)
    
    # Load dataset
    print("Loading dataset...")
    current_dir = os.getcwd()
    os.chdir(dataset_dir)
    ds = load_dataset(dataset_config['path'])
    os.chdir(current_dir)
    ds = ds[split]
    

    len_ds = len(ds)
    if len_ds < n_samples:
        print(f"Warning: Only {len_ds} images available, sampling all of them")
        n_samples = len_ds
    
    # Randomly sample indices
    sampled_indices = random.sample(range(len_ds), n_samples)

    # Create output directory
    output_dir = os.path.join(current_dir, dataset_name, split, f"sampled_images_seed_{seed}_n_{n_samples}")
    os.makedirs(output_dir, exist_ok=True)

    save_images_and_metadata(ds, sampled_indices, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample random images from dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args.config) 