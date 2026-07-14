from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

"""
Extracts per-image CDAL-CS contextual-diversity features from a trained
ssd.pytorch checkpoint, for use with CDAL's CDAL_CS.py selector.
"""
def compute_cdal_context(conf_logits, eps=1e-6):
    """
    Compute contextual features exactly as required by the CDAL selector.

    Parameters
    ----------
    conf_logits : Tensor
        Shape:
            (R, C)      single image
            (B, R, C)   batch

        Raw class logits from SSD.

    Returns
    -------
    Tensor
        Shape:
            (C, C)      single image
            (B, C, C)   batch
    """
    if conf_logits.dim() == 2:
        conf_logits = conf_logits.unsqueeze(0)

    B, R, C = conf_logits.shape

    prob = F.softmax(conf_logits, dim=-1)
    entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)
    pred = prob.argmax(dim=-1)

    context = []
    for b in range(B):
        matrix = torch.zeros((C, C), device=prob.device, dtype=prob.dtype)
        for cls in range(C):
            mask = pred[b] == cls
            if mask.sum() == 0:
                continue
            p = prob[b][mask]
            w = entropy[b][mask]
            matrix[cls] = (p * w.unsqueeze(1)).sum(0) / (w.sum() + eps)
        context.append(matrix)

    return torch.stack(context)


def save_context_features(features, image_ids, output_dir):
    """
    Save one contextual feature matrix per image.

    Parameters
    ----------
    features : Tensor (B,C,C)
    image_ids : list[str]
    output_dir : str
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = features.cpu().numpy().astype(np.float32)

    for feat, img_id in zip(features, image_ids):
        np.save(output_dir / f"{img_id}.npy", feat)