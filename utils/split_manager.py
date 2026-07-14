"""
split_manager.py

Utilities for creating and updating Active Learning splits.
"""

import random
import shutil

from utils.dataset import save_split, load_split

class SplitManager:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

    def create_initial_split(self, image_pool, budget):
        image_pool = list(image_pool)
        random.shuffle(image_pool)

        train = image_pool[:budget]
        unlabeled = image_pool[budget:]

        return train, unlabeled

    def update_split(self, train, unlabeled, selected):
        train = list(train)
        unlabeled = list(unlabeled)

        selected = set(selected)

        train.extend(selected)

        unlabeled = [img for img in unlabeled if img not in selected]

        return train, unlabeled

    def load_split(self, file_path):
        return load_split(file_path)

    def save_split(self, images, file_path):
        save_split(images, file_path)

    def copy_split(self, src, dst):
        shutil.copy(src, dst)

    def summary(self, train, unlabeled):
        total = len(train) + len(unlabeled)

        print("=" * 50)
        print(f"Total images : {total}")
        print(f"Labeled   : {len(train)}")
        print(f"Unlabeled : {len(unlabeled)}")
        print(f"Labeled %    : {100 * len(train) / total:.2f}%")
        print("=" * 50)
        