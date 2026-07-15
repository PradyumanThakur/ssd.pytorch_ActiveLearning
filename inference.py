"""
inference.py

Extract contextual CDAL features from a trained SSD model.

Outputs:
experiments/
    VOC_SSD300_42/
        cdal/
            round_01/
                inference/
                    labeled/
                        xxxx.npy
                    unlabeled/
                        xxxx.npy
"""

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from tqdm import tqdm

from data import (
    VOC_ROOT,
    VOCDetection,
    BaseTransform,
    VOCAnnotationTransform,
    inference_collate,
    VOC_CLASSES,
)

from data.config import voc
from ssd import build_ssd
from utils.inference_utils import compute_cdal_context, save_context_features

MEANS = (104, 117, 123)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="SSD CDAL Feature Extraction")

parser.add_argument("--dataset_root", default=VOC_ROOT, type=str)
parser.add_argument("--split-file", required=True, type=str, help="train.txt / unlabeled.txt")
parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint_best.pth or checkpoint_latest.pth")
parser.add_argument("--output-dir", required=True, type=str, help="directory to save contextual features")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--cuda", default=True, type=str2bool)

args = parser.parse_args()


device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


dataset = VOCDetection(
    root=args.dataset_root,
    split_file=args.split_file,
    transform=BaseTransform(
        300,
        MEANS,
    ),
    target_transform=VOCAnnotationTransform(),
)

loader = data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    collate_fn=inference_collate,
    pin_memory=True,
    persistent_workers=args.workers > 0,
)

print("=" * 60)
print("Building SSD...")
print("=" * 60)

ssd_net = build_ssd(phase="train", size=300, num_classes=voc["num_classes"])

checkpoint = torch.load(args.checkpoint, map_location=device)

# checkpoint saved from train.py
if "model" in checkpoint:
    ssd_net.load_state_dict(checkpoint["model"])
else:
    ssd_net.load_state_dict(checkpoint)

ssd_net = ssd_net.to(device)
ssd_net.eval()

print("Checkpoint loaded:")
print(args.checkpoint)

print()
print("Dataset size :", len(dataset))
print("Batch size   :", args.batch_size)
print("Output dir   :", args.output_dir)
print("=" * 60)


def inference():
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\nStarting feature extraction...\n")

    n_saved = 0
    n_skipped = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for batch in pbar:
            images, _, image_ids = batch
            images = images.to(device, non_blocking=True)

            loc, conf, priors = ssd_net(images)

            # Remove background column
            # (B, R, 21) - > (B, R, 20)
            conf = conf[..., 1:]

            # Compute contextual feature
            # Input: (B, R, C)
            # Output: (B, C, C)
            context = compute_cdal_context(conf)

            # Save one feature per image
            save_context_features(features=context, image_ids=image_ids, output_dir=args.output_dir)

            n_saved += len(image_ids)

            pbar.set_postfix(saved=n_saved)
        
    print()
    print("=" * 60)
    print("Inference Finished")
    print("=" * 60)
    print(f"Saved features : {n_saved}")
    print(f"Output folder  : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    inference()
