import os
import yaml
import json
from datasets import load_dataset, Image
from google import genai
from dotenv import load_dotenv
load_dotenv()

def load_client(model_name):
    if "gemma" in model_name or "gemini" in model_name:
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Model {model_name} not supported")
    
def generate_solution(client, model_name, images: list, question: str):
    if "gemma" in model_name or "gemini" in model_name:
        return generate_solution_google(client, model_name, images, question)
    else:
        raise ValueError(f"Model {model_name} not supported yet. Please add support for this model.")

def generate_solution_google(client, model_name, images: list, question: str):
    response = client.models.generate_content(
        model=model_name,
        contents=images + [question],
    )
    return response.text

def count_tokens(client, model_name, prompt):
    if "gemma" in model_name or "gemini" in model_name:
        return count_tokens_google(client, model_name, prompt)
    else:
        raise ValueError(f"Model {model_name} not supported")

def count_tokens_google(client, model_name, prompt):
        response = client.models.count_tokens(
            model=model_name,
            contents=prompt,
        )
        return response.total_tokens

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

def MathVista_make_hint(problem):
    qt = problem.get("question_type")
    at = problem.get("answer_type")
    precision = problem.get("precision", None)

    if qt == "multi_choice":
        return "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    else:
        if at == "integer":
            return "Hint: Please answer with an integer and provide the final value at the end."
        if at == "float":
            if precision == 1:
                return "Hint: Please answer with a floating-point number with one decimal place (e.g., 1.2) and provide the final value at the end."
            if precision == 2:
                return "Hint: Please answer with a floating-point number with two decimal places (e.g., 1.23) and provide the final value at the end."
            return "Hint: Please answer with a floating-point number and provide the final value at the end."
        if at == "list":
            return "Hint: Please answer with a Python list (e.g., [1, 2, 3]) and provide the final list at the end."
        return "Hint: Please provide the final answer clearly at the end."

def MathVista_make_prompt(problem):
    # Question (with optional unit)
    q = problem.get("question", "").strip()
    unit = problem.get("unit")
    question_line = f"Question: {q}"
    if unit and str(unit).lower() != "none":
        question_line += f" (Unit: {unit})"

    # Choices (if any)
    choices = problem.get("choices")
    if choices and str(choices).lower() != "none":
        lines = ["Choices:"]
        for i, choice in enumerate(choices):
            lines.append(f"({chr(ord('A')+i)}) {choice}")
        choices_block = "\n".join(lines)
    else:
        choices_block = ""

    hint = MathVista_make_hint(problem)

    # Always solution type
    tail = "Always end your solution with the final answer in latex, using '\\boxed{{<answer>}}'."

    elements = [question_line, choices_block, hint, tail]
    return "\n".join([e for e in elements if e]).strip()

def save_json(obj, path, append=False):
    if append:
        with open(path, "a", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)