import os
import numpy as np
from dataclasses import dataclass

def load_features(feature_dir, split_file):
    image_ids = []
    features = []

    with open(split_file) as f:
        for line in f:
            dataset, image_id = line.strip().split()
            feature = np.load(os.path.join(feature_dir, image_id + ".npy"))

            features.append(feature.reshape(-1))
            image_ids.append((dataset, image_id))

    return np.stack(features), image_ids


def _kl_distance_matrix(A, B, num_classes, chunk_size=64, eps=1e-10):
    """
    A: [N, C*C], B: [M, C*C]
    returns: [N, M] fully vectorized symmetric KL distance matrix
    Matches sym_kl_divergence logic: per-class KL, NaN rows dropped, average over valid classes
    """
    N = len(A)
    M = len(B)

    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
                   
    Ar = A.reshape(N, num_classes, num_classes) # [N, C, C]
    Br = B.reshape(M, num_classes, num_classes) # [M, C, C]
    D  = np.empty((N, M), dtype=np.float32)

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)

        pa = Ar[start:stop, None]      # [chunk,1,C,C]
        pb = Br[None]                  # [1,M,C,C]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_pa_pb = np.log(pa / pb)
            log_pb_pa = np.log(pb / pa)
            
            # per-class symmetric KL: [chunk, M, C]        
            kl = (-0.5 * pa * log_pa_pb - 0.5 * pb * log_pb_pa).sum(axis=-1)
        
        # drop NaN/Inf rows exactly like the original, then average
        kl = np.where(np.isfinite(kl), kl, np.nan)          # [chunk, M, C]
        with np.errstate(all='ignore'):
            kl_mean = np.nanmean(kl, axis=-1)                # [chunk, M]
        kl_mean = np.where(np.isnan(kl_mean), 0.0, kl_mean) # all-NaN → 0

        D[start:stop] = np.abs(kl_mean).astype(np.float32)

    return D


def cdal_coreset_select(unlabeled_features, labeled_features,
                        budget, num_classes, seed=None):
    """
    Precomputes ALL pairwise distances upfront — greedy loop is then O(N) lookups.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    unlabeled_features = np.asarray(unlabeled_features, dtype=np.float32)
    labeled_features   = np.asarray(labeled_features,   dtype=np.float32)

    N_u = len(unlabeled_features)

    print("Computing distances to labeled...")
    dist_ul = _kl_distance_matrix(
        unlabeled_features, labeled_features, num_classes=num_classes
    )   # [N_u, N_l]

    min_dist = dist_ul.min(axis=1)   # [N_u]

    selected = []

    for idx in range(min(budget, N_u)):
        if idx == 0:
            chosen = int(rng.choice(N_u))          # random first pick, matches original
        else:
            chosen = int(np.argmax(min_dist))

        selected.append(chosen)
        min_dist[chosen] = -np.inf                 # mark as selected

        # Compute only ONE column (N_u × 1)
        dist_new = _kl_distance_matrix(
            unlabeled_features,
            unlabeled_features[chosen:chosen + 1],
            num_classes=num_classes,
        ).ravel()

        active = min_dist >= 0
        
        min_dist[active] = np.minimum(
            min_dist[active],
            dist_new[active],
        )

    return selected


@dataclass
class ImageRecord:
    dataset: str
    image_id: str

def update_splits(selected_indices, unlabeled_records,
                  labeled_records, next_round_split_dir, current_round_split_dir=None):
    
    """
    Parameters
    ----------
    selected_indices : list[int]
        Indices selected by CDAL.

    unlabeled_records : list[(dataset, image_id)]

    labeled_records : list[(dataset, image_id)]

    next_round_split_dir : str
        experiments/.../round_xx/splits/

    current_round_split_dir : str or None
        experiments/.../round_xx/splits/
        If given, selected.txt is written here.
    """

    os.makedirs(next_round_split_dir, exist_ok=True)

    selected_records = [unlabeled_records[i] for i in selected_indices]

    new_labeled = labeled_records + selected_records

    selected_set = set(selected_indices)

    new_unlabeled = [record for idx, record in enumerate(unlabeled_records) if idx not in selected_set]

    if current_round_split_dir is not None:
        os.makedirs(current_round_split_dir, exist_ok=True)

        with open(os.path.join(current_round_split_dir, "selected.txt"), "w") as f:
            for record in selected_records:
                dataset = record.dataset
                image_id = record.image_id
                f.write(f"{dataset} {image_id}\n")

    with open(os.path.join(next_round_split_dir, "train.txt"), "w") as f:
        for record in new_labeled:
            dataset = record.dataset
            image_id = record.image_id
            f.write(f"{dataset} {image_id}\n")
    
    with open(os.path.join(next_round_split_dir, "unlabeled.txt"), "w") as f:
        for record in new_unlabeled:
            dataset = record.dataset
            image_id = record.image_id
            f.write(f"{dataset} {image_id}\n")


    return new_labeled, new_unlabeled, selected_records