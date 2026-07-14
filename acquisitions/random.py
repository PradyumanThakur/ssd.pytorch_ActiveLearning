import os
import random


def random_select(unlabeled_records, budget, seed=None):
    """
    Randomly select images from the unlabeled pool.

    Parameters
    ----------
    unlabeled_records : list[(dataset, image_id)]

    budget : int

    seed : int

    Returns
    -------
    selected_indices : list[int]
    """

    rng = random.Random(seed)

    budget = min(budget, len(unlabeled_records))

    selected_indices = rng.sample(range(len(unlabeled_records)), budget)

    return sorted(selected_indices)