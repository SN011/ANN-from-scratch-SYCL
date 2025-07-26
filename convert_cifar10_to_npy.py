import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _discover_classes(root: Path) -> List[str]:
    """Return a sorted list of class names (sub-directory names) under *root*."""
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class sub-directories found in {root}")
    return sorted(classes)


def _gather_image_paths(split_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Recursively collect image paths in *split_dir* that match given *extensions*."""
    image_paths: List[Path] = []
    for ext in extensions:
        image_paths.extend(split_dir.rglob(f"*.{ext}"))
    return sorted(image_paths)


def _load_split(split_root: Path, class_to_idx: Dict[str, int], img_size: int, extensions: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels from *split_root* (train or test)."""
    image_paths = _gather_image_paths(split_root, extensions)
    if not image_paths:
        raise RuntimeError(f"No images with extensions {extensions} found under {split_root}")

    imgs: List[np.ndarray] = []
    labels: List[int] = []

    for img_path in image_paths:
        # The parent directory name corresponds to the class.
        class_name = img_path.parent.name
        if class_name not in class_to_idx:
            raise KeyError(f"Encountered unknown class directory '{class_name}' in {img_path}")
        label = class_to_idx[class_name]

        with Image.open(img_path) as img:
            img = img.convert("RGB")  # ensure 3-channel
            if img_size is not None:
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            imgs.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)

    X = np.stack(imgs, axis=0)  # (N, H, W, 3)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def convert_cifar10_to_npy(dataset_root: Path, output_dir: Path, img_size: int | None, extensions: Tuple[str, ...] = ("png", "jpg", "jpeg")) -> None:
    """Convert CIFAR-10 image folder to NumPy arrays and save them in *output_dir*."""
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"

    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError("Expected 'train' and 'test' sub-directories under " f"{dataset_root}")

    class_names = _discover_classes(train_root)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    print("Discovered classes:")
    for cls_name, idx in class_to_idx.items():
        print(f"  {idx}: {cls_name}")

    print("\nLoading train split ...")
    X_train, y_train = _load_split(train_root, class_to_idx, img_size, extensions)
    print(f"Train: images {X_train.shape}, labels {y_train.shape}")

    print("\nLoading test split ...")
    X_test, y_test = _load_split(test_root, class_to_idx, img_size, extensions)
    print(f"Test: images {X_test.shape}, labels {y_test.shape}")

    # Ensure output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save npy files.
    np.save(output_dir / "train_images.npy", X_train)
    np.save(output_dir / "train_labels.npy", y_train)
    np.save(output_dir / "test_images.npy", X_test)
    np.save(output_dir / "test_labels.npy", y_test)

    print(f"\nSaved NumPy binaries to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CIFAR-10 image folders to .npy binaries.")
    parser.add_argument("dataset_root", type=Path, help="Path to folder containing 'train' and 'test' sub-folders.")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the .npy files.")
    parser.add_argument("--img-size", type=int, default=None, help="Optional: resize all images to this size (default: keep original)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_cifar10_to_npy(args.dataset_root, args.output_dir, args.img_size) 