import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.experiment import ExperimentManager
from utils.split_manager import SplitManager
from utils.pipeline import run_training_pipeline

from acquisitions.random import random_select
from acquisitions.cdal_cs import (
    load_features,
    cdal_coreset_select,
    update_splits,
)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Active Learning round.")

    parser.add_argument("--method", choices=["random", "cdal"], required=True)
    parser.add_argument("--round", type=int, required=True,
                        help="Current AL round.\nExample: round=1 trains Round-01 and creates Round-02.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", default="VOC")
    parser.add_argument("--dataset_root", default="datasets/PASCAL_VOC/VOCdevkit")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--save-freq", default=9999, type=int)
    parser.add_argument("--cuda", default=True, type=str2bool)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--acquisition-budget", default=1000, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    experiment = ExperimentManager(args.experiment_name)
    split_manager = SplitManager(seed=args.seed)

    current_round = args.round
    next_round = current_round + 1

    print("=" * 70)
    print(f"Running {args.method.upper()} Round-{current_round:02d}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load current split
    # ----------------------------------------------------------
    train_file = (experiment.get_split_dir(args.method, current_round) / "train.txt")
    unlabeled_file = (experiment.get_split_dir(args.method, current_round) / "unlabeled.txt")

    train = split_manager.load_split(train_file)
    unlabeled = split_manager.load_split(unlabeled_file)

    split_manager.summary(train, unlabeled)

    # ----------------------------------------------------------
    # Create next round directories
    # ----------------------------------------------------------
    experiment.create_method_round(args.method, next_round)

    # ----------------------------------------------------------
    # Train + Evaluate + Feature Extraction
    # ----------------------------------------------------------
    checkpoint, metrics = run_training_pipeline(
        train_file=train_file,
        unlabeled_file=unlabeled_file,
        weight_dir=experiment.get_weight_dir(
            args.method,
            current_round,
        ),
        evaluation_dir=experiment.get_evaluation_dir(
            args.method,
            current_round,
        ),
        inference_dir=experiment.get_inference_dir(
            args.method,
            current_round,
        ),
        args=args,
        
    )

    # ----------------------------------------------------------
    # Save AL history
    # ----------------------------------------------------------
    experiment.append_history(
        experiment.al_history,
        {
            "method": args.method,
            "round": current_round,
            "labeled": len(train),
            "unlabeled": len(unlabeled),
            "mAP": metrics["mAP"],
        },
    )

    experiment.append_history(
        experiment.per_class_ap,
            {
            "method": args.method,
            "round": current_round,
            **metrics["aps"]
        },
    )

    # ----------------------------------------------------------
    # Load contextual features
    # ----------------------------------------------------------
    print("\nLoading contextual features...\n")

    labeled_features, _ = load_features(
        feature_dir=experiment.get_inference_dir(
            args.method,
            current_round,
        ) / "labeled",
        split_file=train_file
    )

    unlabeled_features, _ = load_features(
        feature_dir=experiment.get_inference_dir(
            args.method,
            current_round,
        ) / "unlabeled",
        split_file=unlabeled_file
    )

    # ----------------------------------------------------------
    # Acquisition
    # ----------------------------------------------------------
    print(f"\nRunning {args.method.upper()} acquisition...\n")

    if args.method == "random":
        selected_indices = random_select(
            unlabeled_records=unlabeled,
            budget=args.acquisition_budget,
            seed=args.seed,
        )
    elif args.method == "cdal":
        selected_indices = cdal_coreset_select(
            unlabeled_features=unlabeled_features,
            labeled_features=labeled_features,
            budget=args.acquisition_budget,
            num_classes=20,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown acquisition method : {args.method}")

    # ----------------------------------------------------------
    # Update splits
    # ----------------------------------------------------------
    update_splits(
        selected_indices=selected_indices,
        unlabeled_records=unlabeled,
        labeled_records=train,
        next_round_split_dir=experiment.get_split_dir(
            args.method,
            next_round,
        ),
        current_round_split_dir=(
            experiment.get_selected_dir(
                args.method,
                current_round,
            )
        ),
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n")
    print("=" * 70)
    print(f"{args.method.upper()} Round-{current_round:02d} Finished")
    print("=" * 70)

    print(f"Labeled Images      : {len(train)}")
    print(f"Unlabeled Images    : {len(unlabeled)}")
    print(f"Acquired Images     : {len(selected_indices)}")
    print(f"Current mAP         : {metrics['mAP']:.4f}")

    print("\nNext Round")
    print(f"Round               : {next_round:02d}")
    print(
        f"Train Split         : "
        f"{experiment.get_split_dir(args.method, next_round) / 'train.txt'}"
    )
    print(
        f"Unlabeled Split     : "
        f"{experiment.get_split_dir(args.method, next_round) / 'unlabeled.txt'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()