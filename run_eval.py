import os
import sys
import json
import argparse
from extract_utils import extract_answer
from utils import *

def main(args):
    config = load_config(args.config)
    dataset_name = config['dataset_name']
    split = config['split']
    model_name = config['model']
    
    current_dir = os.getcwd()
    solutions_dir = os.path.join(current_dir, dataset_name, split, f"generated_solutions_{model_name}")
    solutions_path = os.path.join(solutions_dir, f"solutions_{dataset_name}_{split}.json")

    # Load solutions: expected format { pid: solution_text }
    solutions = load_json(solutions_path) # { pid: solution_text }

    output_path = os.path.join(solutions_dir, f"answers_{dataset_name}_{split}.json")

    # Extract answers
    answers = {}
    num_none = 0
    for pid, solution_text in solutions.items():
        ans = extract_answer(solution_text)
        if ans is None:
            num_none += 1
        answers[pid] = ans

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(answers, output_path)

    total = len(solutions)
    extracted = total - num_none
    print(f"Wrote answers to: {output_path}")
    print(f"Total: {total} | Extracted: {extracted} | Missing: {num_none}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract final answers from solutions JSON (expects \\boxed{...} at end of solutions).")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args)

