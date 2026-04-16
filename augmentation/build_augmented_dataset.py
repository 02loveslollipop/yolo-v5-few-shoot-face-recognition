#!/usr/bin/env python3
"""Create an augmented YOLO dataset from the reviewed base dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml


PAD_FILL = (114, 114, 114)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_augmented"))
    parser.add_argument("--augmentations-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_classes(classes_path: Path) -> list[str]:
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_path}")
    return classes


def reset_output_dir(output_dir: Path) -> None:
    for relative in ("images", "labels"):
        shutil.rmtree(output_dir / relative, ignore_errors=True)
    for relative in ("dataset.yaml", "classes.txt", "summary.json"):
        target = output_dir / relative
        if target.exists():
            target.unlink()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_label_file(label_path: Path) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    bboxes: list[tuple[float, float, float, float]] = []
    class_labels: list[int] = []
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label line in {label_path}: {raw_line}")
        class_labels.append(int(parts[0]))
        bboxes.append(tuple(float(value) for value in parts[1:5]))
    return bboxes, class_labels


def write_label_file(label_path: Path, class_labels: list[int], bboxes: list[tuple[float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, bbox in zip(class_labels, bboxes):
        x_center, y_center, width, height = bbox
        lines.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), encoded, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise OSError(f"Failed to write image: {image_path}")


def build_transform(seed: int) -> A.Compose:
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Affine(
                        scale=(0.82, 1.18),
                        translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                        rotate=(-18, 18),
                        shear=(-8, 8),
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=PAD_FILL,
                        p=1.0,
                    ),
                    A.Affine(
                        scale=(1.0, 1.0),
                        translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                        rotate=(-16, 16),
                        shear=(0.0, 0.0),
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=PAD_FILL,
                        p=1.0,
                    ),
                ],
                p=0.95,
            ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=(-18, 18),
                sat_shift_limit=(-35, 35),
                val_shift_limit=(-28, 28),
                p=0.85,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.22, 0.28),
                contrast_limit=(-0.2, 0.2),
                p=0.75,
            ),
            A.ToGray(p=0.2),
            A.OneOf(
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 6),
                        hole_height_range=(0.05, 0.18),
                        hole_width_range=(0.05, 0.18),
                        fill=0,
                        p=1.0,
                    ),
                    A.GridDropout(
                        ratio=0.35,
                        unit_size_range=(40, 96),
                        fill=0,
                        p=1.0,
                    ),
                    A.PixelDropout(
                        dropout_prob=0.02,
                        per_channel=False,
                        drop_value=0,
                        p=1.0,
                    ),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                    A.Perspective(
                        scale=(0.02, 0.05),
                        keep_size=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=PAD_FILL,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.35,
            clip=True,
            filter_invalid_bboxes=True,
        ),
        seed=seed,
    )


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
    }
    with (output_dir / "dataset.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n")


def collect_summary(
    output_dir: Path,
    classes: list[str],
    augmentations_per_image: int,
    seed: int,
) -> dict[str, object]:
    summary = {
        "classes": classes,
        "augmentations_per_image": augmentations_per_image,
        "seed": seed,
        "split_image_counts": {},
        "split_label_counts": {},
        "instance_counts": {},
    }
    instance_counts = Counter()
    for split in ("train", "val"):
        image_count = len(list((output_dir / "images" / split).glob("*.jpg")))
        label_paths = list((output_dir / "labels" / split).glob("*.txt"))
        summary["split_image_counts"][split] = image_count
        summary["split_label_counts"][split] = len(label_paths)
        for label_path in label_paths:
            _, class_labels = read_label_file(label_path)
            for class_id in class_labels:
                instance_counts[classes[class_id]] += 1
    summary["instance_counts"] = dict(instance_counts)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_dir = args.input_dir
    output_dir = args.output_dir
    classes = read_classes(input_dir / "classes.txt")
    transform = build_transform(args.seed)

    reset_output_dir(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        image_dir = input_dir / "images" / split
        label_dir = input_dir / "labels" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}")

            copy_file(image_path, output_dir / "images" / split / image_path.name)
            copy_file(label_path, output_dir / "labels" / split / label_path.name)

            if split != "train":
                continue

            image = read_image(image_path)
            bboxes, class_labels = read_label_file(label_path)
            if not bboxes:
                continue
            for aug_index in range(args.augmentations_per_image):
                transformed_payload = None
                for _ in range(args.max_attempts):
                    candidate = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    if candidate["bboxes"]:
                        transformed_payload = candidate
                        break
                if transformed_payload is None:
                    raise RuntimeError(
                        f"Failed to generate a valid augmented sample for {image_path.name} "
                        f"after {args.max_attempts} attempts."
                    )

                aug_image_name = f"{image_path.stem}__aug_{aug_index + 1:02d}.jpg"
                aug_label_name = f"{image_path.stem}__aug_{aug_index + 1:02d}.txt"
                write_image(output_dir / "images" / "train" / aug_image_name, transformed_payload["image"])
                write_label_file(
                    output_dir / "labels" / "train" / aug_label_name,
                    list(transformed_payload["class_labels"]),
                    list(transformed_payload["bboxes"]),
                )

    write_dataset_yaml(output_dir, classes)
    summary = collect_summary(output_dir, classes, args.augmentations_per_image, args.seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Built augmented dataset at {output_dir} with "
        f"{summary['split_image_counts']['train']} train images and "
        f"{summary['split_image_counts']['val']} val images."
    )


if __name__ == "__main__":
    main()
