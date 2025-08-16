import os
import argparse
from datasets import load_dataset
from utils import *
import string

def main(args):
    config = load_config(args.config)
    dataset_name = config['dataset_name']
    split = config['split']
    model_name = config['model']

    current_dir = os.getcwd()

    # Where answers are written by run_eval.py
    solutions_dir = os.path.join(current_dir, dataset_name, split, f"generated_solutions_{model_name}")
    answers_path = os.path.join(solutions_dir, f"answers_{dataset_name}_{split}.json")

    # Load predictions: { pid: answer or None }
    preds = load_json(answers_path)

    # Load ground-truth dataset
    dataset_cfg = load_dataset_config(dataset_name, split)
    dataset_path = dataset_cfg['path']
    ds = load_dataset(dataset_path)[split]

    # Build pid -> gt map (using 'answer')
    gt = {}
    for i, row in enumerate(ds):
        pid = str(row['pid'])
        ans = str(row['answer']).strip()
        if row['question_type'] == 'multi_choice': 
            # i.e. answer looks like this: (A) larger than, we want to extract A given ans = larger than
            if ans in row['choices']:
                idx = row['choices'].index(ans)
                letter = string.ascii_uppercase[idx]  # 0→A, 1→B, 2→C...
                ans = letter
        gt[pid] = ans

    # Compare
    total = len(gt)
    correct = 0
    missing = 0
    incorrect = 0

    for pid, gt_ans in gt.items():
        if pid not in preds:
            missing += 1
            continue
        pred_ans = preds[pid]
        if pred_ans is None:
            incorrect += 1
            continue
        if str(pred_ans).strip() == gt_ans:
            correct += 1
        else:
            incorrect += 1

    extracted = len([v for v in preds.values() if v is not None])
    print(f"Compared answers: {answers_path}")
    print(f"Total: {total} | Correct: {correct} | Incorrect: {incorrect} | Missing: {missing} | Extracted: {extracted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directly compare answers to dataset ground truth (field 'answer').")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)


