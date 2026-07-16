import json
import subprocess
from pathlib import Path

import torch
import pandas as pd

from ssd import build_ssd
from data.config import voc

from eval import evaluate_model


def run_command(cmd):
    """
    Execute a subprocess while printing the command.

    Raises
    ------
    CalledProcessError
        if the command fails.
    """

    print("\n" + "=" * 70)
    print("Running:")
    print(" ".join(map(str, cmd)))
    print("=" * 70)

    subprocess.run(cmd, check=True)


def train_model(train_file, weight_dir, args):
    """
    Train SSD on the current labeled split.

    Returns
    -------
    Path
        checkpoint_best.pth
    """
    cmd = [
        "python", "train.py",
        "--dataset", args.dataset,
        "--dataset_root", args.dataset_root,
        "--split-file", str(train_file),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--save-dir", str(weight_dir),
        "--save-freq", str(args.save_freq),
        "--experiment-name", args.experiment_name
    ]

    if args.resume is not None:
        cmd.extend(["--resume", args.resume])

    if args.cuda:
        cmd.extend(["--cuda", "True"])

    run_command(cmd)

    checkpoint = Path(weight_dir) / "checkpoint_best.pth"

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    return checkpoint


def evaluate_round(checkpoint, dataset_root, evaluation_dir, device):
    """
    Evaluate checkpoint_best.pth on VOC2007 test.
    """
    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(checkpoint)

    # Build SSD model (same architecture used during training)
    model = build_ssd(phase="train", size=300, num_classes=voc["num_classes"])

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    # Evaluate
    metrics = evaluate_model(
        net=model,
        dataset_root=dataset_root,
        device=device,
        save_dir=evaluation_dir,
    )

    return metrics

    
def extract_features(checkpoint, split_file, output_dir, dataset_root, cuda=True, batch_size=32, num_workers=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_cmd = [
        "python", "inference.py",
        "--checkpoint", str(checkpoint),
        "--dataset_root", dataset_root,
        "--split-file", str(split_file),
        "--output-dir", str(output_dir),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
    ]

    if cuda:
        inference_cmd.extend(["--cuda", "True"])

    run_command(inference_cmd)


def save_round_metrics(metrics, evaluation_dir):
    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = evaluation_dir / "metrics.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to {metrics_file}")


def run_training_pipeline(train_file, unlabeled_file, weight_dir, evaluation_dir, inference_dir, args):

    device = torch.device(
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "cpu"
    )

    print("\n")
    print("=" * 70)
    print("=" * 70)

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    checkpoint = train_model(
        train_file=train_file,
        weight_dir=weight_dir,
        args=args,
    )

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------

    if metrics_exist(evaluation_dir):
        print("\nEvaluation already exists. Skipping...")
        metrics = load_metrics(evaluation_dir)
    else:
        metrics = evaluate_round(
            checkpoint=checkpoint,
            dataset_root=args.dataset_root,
            evaluation_dir=evaluation_dir,
            device=device,
        )

        save_round_metrics(metrics, evaluation_dir)

    # -------------------------------------------------------------
    # Labeled features
    # -------------------------------------------------------------

    print("\nExtracting labeled contextual features...\n")

    labeled_dir = Path(inference_dir) / "labeled"

    if features_exist(train_file, labeled_dir):
        print("\nLabeled features already exist. Skipping inference.")
    else:
        print("\nExtracting labeled contextual features...\n")

        extract_features(
            checkpoint=checkpoint,
            split_file=train_file,
            output_dir=labeled_dir,
            dataset_root=args.dataset_root,
            cuda=args.cuda,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # -------------------------------------------------------------
    # Unlabeled features
    # -------------------------------------------------------------

    print("\nExtracting unlabeled contextual features...\n")

    unlabeled_dir = Path(inference_dir) / "unlabeled"

    if features_exist(unlabeled_file, unlabeled_dir):
        print("\nUnlabeled features already exist. Skipping inference.")
    else:
        print("\nExtracting unlabeled contextual features...\n")

        extract_features(
            checkpoint=checkpoint,
            split_file=unlabeled_file,
            output_dir=unlabeled_dir,
            dataset_root=args.dataset_root,
            cuda=args.cuda,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print("\n")
    print("=" * 70)
    print("=" * 70)

    return checkpoint, metrics

<<<<<<< HEAD
def metrics_exist(evaluation_dir):
    return (Path(evaluation_dir) / "metrics.json").exists()


def load_metrics(evaluation_dir):
    with open(Path(evaluation_dir) / "metrics.json", "r") as f:
        return json.load(f)
    
def features_exist(split_file, output_dir):
    """
    Returns True if every image in split_file already has a .npy feature.
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return False

    with open(split_file) as f:
        for line in f:
            _, image_id = line.strip().split()

            if not (output_dir / f"{image_id}.npy").exists():
                return False

    return True
=======
>>>>>>> a4ec4be01f388dfc8cfcea918b80b13de95b61a8
