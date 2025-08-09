import os
from datasets import load_dataset, Image
import argparse
import random
import numpy as np
import yaml

def load_dataset_config(dataset_name, subset):
    """Load dataset configuration from datasets.yaml"""

    with open('datasets.yaml', 'r') as f:
        datasets = yaml.safe_load(f)

    assert dataset_name in datasets['datasets'], f"Dataset name {dataset_name} not found in datasets.yaml"

    dataset_config = datasets['datasets'][dataset_name]
    assert subset in dataset_config['subsets'], f"Subset {subset} not found in datasets.yaml"

    return dataset_config

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_images_and_metadata(ds, sampled_indices, output_dir):
    """Save sampled images and metadata to visualize"""
    
    # Cast image column to decode images if needed
    if 'decoded_image' not in ds.features:
        ds_with_images = ds.cast_column("image", Image(decode=True))
    else:
        ds_with_images = ds
    
    print(f"\nSaving {len(sampled_indices)} images to: {output_dir}")
    
    for i, idx in enumerate(sampled_indices, 1):
        example = ds_with_images[idx]
        
        # Get the PIL image - try decoded_image first, then image
        if 'decoded_image' in example and example['decoded_image'] is not None:
            pil_image = example['decoded_image']
        else:
            pil_image = example['image']
        
        # Create filename using problem id
        filename = f"sample_image_index_{idx}.png"
        save_path = os.path.join(output_dir, filename)
        
        pil_image.save(save_path)
        print(f"{i}/{len(sampled_indices)}: Saved {filename}")

    
    print(f"Complete! Saved {len(sampled_indices)} images to: {output_dir}")

    # Save list of sampled indices
    metadata_path = os.path.join(output_dir, "sampled_indices.txt")
    with open(metadata_path, 'w') as f:
        for idx in sampled_indices:
            f.write(f"{idx}\n")

def main(config_path):
    """Sample random images from dataset and save them"""

    config = load_config(config_path)
    dataset_dir = config['dataset_dir']
    dataset_name = config['dataset_name']
    subset = config['subset']
    n_samples = config['n_samples']
    seed = config['seed']

    # Get dataset config and config file
    dataset_config = load_dataset_config(dataset_name, subset)
    
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
    ds = ds[subset]
    

    len_ds = len(ds)
    if len_ds < n_samples:
        print(f"Warning: Only {len_ds} images available, sampling all of them")
        n_samples = len_ds
    
    # Randomly sample indices
    sampled_indices = random.sample(range(len_ds), n_samples)

    # Create output directory
    output_dir = os.path.join(current_dir, f"sampled_images_seed_{seed}_n_{n_samples}")
    os.makedirs(output_dir, exist_ok=True)

    save_images_and_metadata(ds, sampled_indices, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample random images from dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args.config) 