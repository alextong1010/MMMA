import os
import yaml
from datasets import load_dataset, Image
from google import genai
from dotenv import load_dotenv
load_dotenv()

def load_client(model_name):
    if model_name in ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]:
        return genai.Client(api_key=os.getenv("GEMMA_API_KEY"))
    else:
        raise ValueError(f"Model {model_name} not supported")

def generate_solution(client, model_name, images: list, question: str):
    response = client.models.generate_content(
        model=model_name,
        contents=images + [question],
    )
    return response.text

def load_dataset_config(dataset_name, split):
    """Load dataset configuration from datasets.yaml"""

    with open('datasets.yaml', 'r') as f:
        datasets = yaml.safe_load(f)

    assert dataset_name in datasets['datasets'], f"Dataset name {dataset_name} not found in datasets.yaml"

    dataset_config = datasets['datasets'][dataset_name]
    assert split in dataset_config['splits'], f"Split {split} not found in datasets.yaml"

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