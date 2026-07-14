"""
run_round_00.py

Runs the shared initial Active Learning round.

Pipeline
--------
1. Randomly initialize labeled pool
2. Train SSD
3. Evaluate on VOC2007 test
4. Extract contextual features
5. Random acquisition
6. CDAL acquisition
"""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.experiment import ExperimentManager
from utils.split_manager import SplitManager
from utils.dataset import collect_trainval

from acquisitions.random import random_select
from acquisitions.cdal_cs import (load_features, cdal_coreset_select, update_splits)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", default=100, type=int,
                    help="Number of training epochs")
    parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
    parser.add_argument("--dataset_root", type=str, default="datasets/PASCAL_VOC/VOCdevkit")
    parser.add_argument("--initial-budget", type=int, default=1000)
    parser.add_argument("--acquisition-budget", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=20, 
                    help="Save checkpoint every N epochs")
    parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
    parser.add_argument("--experiment-name", required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    
    experiment = ExperimentManager(args.experiment_name)

    experiment.create_experiment()
    experiment.create_initial_round()

    round_dir = experiment.get_round_dir("initial", 0)

    # Initial random split
    print("=" * 60)
    print("Creating initial split")
    print("=" * 60)

    dataset = collect_trainval(args.dataset_root)

    split_manager = SplitManager(seed=args.seed)
    train, unlabeled = split_manager.create_initial_split(image_pool=dataset, budget=args.initial_budget)

    split_dir = experiment.get_split_dir("initial", 0)

    train_file = split_dir / "train.txt"
    unlabeled_file = split_dir / "unlabeled.txt"

    split_manager.save_split(train, train_file)
    split_manager.save_split(unlabeled, unlabeled_file)

    split_manager.summary(train, unlabeled)

    print("\nExperiment created successfully at: ", end="")
    print(experiment.exp_dir)


    # Train SSD
    print("\nTraining Round 00...\n")
    train_cmd = [
        "python", "train.py",
        "--dataset", "VOC",
        "--dataset_root", args.dataset_root,
        "--split-file", str(train_file),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--experiment-name", args.experiment_name,
        "--save-dir", str(experiment.get_weight_dir("initial", 0)),
        "--save-freq", str(args.save_freq),
        "--lr", str(args.lr)
    ]

    if args.cuda:
        train_cmd.extend(["--cuda", str(True)])

    subprocess.run(train_cmd, check=True)

    # Inference (labeled)
    print("\nExtracting labeled contextual features...\n")
    subprocess.run([
        "python", "inference.py",
        "--checkpoint", str(experiment.get_weight_dir("initial", 0) / "checkpoint_best.pth"),
        "--dataset_root", args.dataset_root,
        "--split-file", str(train_file),
        "--output-dir", str(experiment.get_inference_dir("initial", 0) / "labeled"),
        "--cuda", str(True)
    ], check=True)

    # Inference (unlabeled)
    print("\nExtracting unlabeled contextual features...\n")
    subprocess.run([
        "python", "inference.py",
        "--checkpoint", str(experiment.get_weight_dir("initial", 0) / "checkpoint_best.pth"),
        "--dataset_root", args.dataset_root,
        "--split-file", str(unlabeled_file),
        "--output-dir", str(experiment.get_inference_dir("initial", 0) / "unlabeled"),
        "--cuda", str(True)
    ], check=True)

    # create Round-01 directories
    experiment.create_method_round("random", 1)
    experiment.create_method_round("cdal", 1)

    # Random acquisition
    print("\nRunning Random acquisition...\n")

    random_indices = random_select(
        unlabeled_records=unlabeled,
        budget=args.acquisition_budget,
        seed=args.seed,
    )

    update_splits(
        selected_indices=random_indices,
        unlabeled_records=unlabeled,
        labeled_records=train,
        next_round_split_dir=experiment.get_split_dir("random", 1),
        current_round_split_dir=(experiment.get_selected_dir("initial", 0) / "random")
    )

    # CDAL acquisition
    print("\nRunning CDAL acquisition...\n")
    labeled_features, _ = load_features(
        feature_dir=experiment.get_inference_dir("initial", 0) / "labeled",
        split_file=train_file,
        num_classes=20,
    )

    unlabeled_features, _ = load_features(
        feature_dir=experiment.get_inference_dir("initial", 0) / "unlabeled",
        split_file=unlabeled_file,
        num_classes=20,
    )

    cdal_indices = cdal_coreset_select(
        unlabeled_features=unlabeled_features,
        labeled_features=labeled_features,
        budget=args.acquisition_budget,
        num_classes=20,
        seed=args.seed,
    )

    update_splits(
        selected_indices=cdal_indices,
        unlabeled_records=unlabeled,
        labeled_records=train,
        next_round_split_dir=experiment.get_split_dir("cdal", 1),
        current_round_split_dir=(
            experiment.get_selected_dir("initial", 0) / "cdal"
        ),
    )

    print("\n" + "=" * 60)
    print("Round-00 completed successfully.")
    print("=" * 60)
    print(f"Initial labeled images : {len(train)}")
    print(f"Initial unlabeled      : {len(unlabeled)}")
    print(f"Random selected        : {len(random_indices)}")
    print(f"CDAL selected          : {len(cdal_indices)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
