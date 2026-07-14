"""
dataset.py

Pascal VOC dataset and split utilities.
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    image_id: str

def collect_trainval(voc_root):
    """
    Collect all images from VOC2007 + VOC2012 trainval.

    Returns
    -------
    List[ImageRecord]
    """
    voc_root = Path(voc_root)
    image_list = []
    datasets = ["VOC2007", "VOC2012"]

    for dataset in datasets:
        txt = (voc_root / dataset / "ImageSets" / "Main" / "trainval.txt")

        if not txt.exists():
            raise FileNotFoundError(txt)

        with open(txt) as f:
            ids = [x.strip() for x in f.readlines()]

        image_list.extend([ImageRecord(dataset, img_id) for img_id in ids])

    return image_list

def collect_test(voc_root):
    """
    Collect VOC2007 test split.

    Used for evaluation.
    """
    voc_root = Path(voc_root)
    txt = (voc_root / "VOC2007" / "ImageSets" / "Main" / "test.txt")

    image_list = []
    with open(txt) as f:
        ids = [x.strip() for x in f.readlines()]

    image_list.extend([ImageRecord("VOC2007", img_id) for img_id in ids])

    return image_list

def load_split(file_path):
    """
    Read a split file.

    Format
    ------
    VOC2007 000001
    VOC2012 2008_000123

    Returns
    -------
    List[ImageRecord]
    """

    file_path = Path(file_path)

    images = []

    with open(file_path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            dataset, image_id = line.split()

            images.append(ImageRecord(dataset, image_id))

    return images


def save_split(images, file_path):
    """
    Save ImageRecord list.

    Format
    ------
    VOC2007 000001
    VOC2012 2008_000123
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:

        for img in images:
            f.write(f"{img.dataset} {img.image_id}\n")