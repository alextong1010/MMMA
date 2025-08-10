import argparse
import json
import os
import yaml
from datasets import load_dataset
from utils import load_config, load_dataset_config


def make_hint(problem):
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

def make_query(problem):
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

    hint = make_hint(problem)

    # Always solution type
    tail = "Always end your solution with the final answer in latex, using '\\boxed{{<answer>}}'."

    elements = [question_line, choices_block, hint, tail]
    return "\n".join([e for e in elements if e]).strip()

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main(config_path):
    config = load_config(config_path)
    dataset_dir = config['dataset_dir']
    dataset_name = config['dataset_name']
    split = config['split']
    dataset_config = load_dataset_config(dataset_name, split)
    dataset_path = dataset_config['path']
    
    # Load dataset
    print("Loading dataset...")
    current_dir = os.getcwd()
    os.chdir(dataset_dir)
    dataset = load_dataset(dataset_path)
    os.chdir(current_dir)

    # Pick split
    split_ds = dataset[split]

    # Create output directory following the same pattern as other scripts
    output_dir = os.path.join(current_dir, dataset_name, split, "generated_queries")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    output_file = os.path.join(output_dir, f"queries_{dataset_name}_{split}.json")

    # Build query dict
    query_data = {}
    for row in split_ds:
        pid = row["pid"]
        query_data[pid] = make_query(row)

    # Save
    save_json(query_data, output_file)
    print(f"Saved {len(query_data)} queries for split '{split}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate queries from dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args.config) 
