#!/usr/bin/env python3
"""Build a YOLOv5-ready dataset from the reviewed annotation manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def clamp_bbox(bbox: tuple[int, int, int, int], size: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(size - 1, x1))
    y1 = max(0, min(size - 1, y1))
    x2 = max(x1 + 1, min(size, x2))
    y2 = max(y1 + 1, min(size, y2))
    return x1, y1, x2, y2


def preprocess_image(src: Path, dst: Path, size: int, pad_color: tuple[int, int, int]) -> Image.Image:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(size / image.width, size / image.height)
        resized_w = max(1, round(image.width * scale))
        resized_h = max(1, round(image.height * scale))
        resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), pad_color)
        offset = ((size - resized_w) // 2, (size - resized_h) // 2)
        canvas.paste(resized, offset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, quality=95)
    return canvas


def yolo_label_line(class_id: int, bbox: tuple[int, int, int, int], size: int) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / size
    y_center = ((y1 + y2) / 2) / size
    width = (x2 - x1) / size
    height = (y2 - y1) / size
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def write_review_image(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    class_name: str,
    dst: Path,
) -> None:
    review = image.copy()
    draw = ImageDraw.Draw(review)
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
    draw.text((x1 + 4, max(0, y1 - 14)), class_name, fill="red")
    dst.parent.mkdir(parents=True, exist_ok=True)
    review.save(dst, quality=92)


def reset_output_dir(output_dir: Path) -> None:
    for relative in ("images", "labels", "review"):
        shutil.rmtree(output_dir / relative, ignore_errors=True)
    for relative in ("dataset.yaml", "classes.txt", "summary.json"):
        target = output_dir / relative
        if target.exists():
            target.unlink()


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n")
    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n")


def build_dataset(
    raw_dir: Path,
    manifest_path: Path,
    classes_path: Path,
    output_dir: Path,
    image_size: int,
) -> None:
    pad_color = (114, 114, 114)
    classes = load_classes(classes_path)
    class_to_id = {name: index for index, name in enumerate(classes)}

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "image_size": image_size,
        "train_images": 0,
        "val_images": 0,
        "excluded_images": 0,
        "classes": classes,
        "class_counts": Counter(),
        "split_counts": Counter(),
    }
    skipped = []

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = raw_dir / row["source"]
            output_name = row["output"]
            class_name = row["class"]
            split = row["split"].strip().lower()
            bbox = clamp_bbox(
                tuple(int(row[key]) for key in ("x1", "y1", "x2", "y2")),
                image_size,
            )

            if split == "exclude":
                summary["excluded_images"] += 1
                skipped.append({"source": row["source"], "reason": row.get("notes", "excluded")})
                continue

            if split not in {"train", "val"}:
                raise ValueError(f"Unsupported split '{split}' in {manifest_path}")
            if class_name not in class_to_id:
                raise ValueError(f"Unknown class '{class_name}' in {manifest_path}")
            if not source.exists():
                raise FileNotFoundError(f"Missing raw image: {source}")

            image_dst = output_dir / "images" / split / output_name
            label_dst = output_dir / "labels" / split / f"{Path(output_name).stem}.txt"
            review_dst = output_dir / "review" / split / output_name

            processed = preprocess_image(source, image_dst, image_size, pad_color)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.write_text(yolo_label_line(class_to_id[class_name], bbox, image_size))
            write_review_image(processed, bbox, class_name, review_dst)

            summary["class_counts"][class_name] += 1
            summary["split_counts"][split] += 1
            summary[f"{split}_images"] += 1

    write_dataset_yaml(output_dir, classes)

    materialized_summary = {
        "image_size": summary["image_size"],
        "train_images": summary["train_images"],
        "val_images": summary["val_images"],
        "excluded_images": summary["excluded_images"],
        "classes": summary["classes"],
        "class_counts": dict(summary["class_counts"]),
        "split_counts": dict(summary["split_counts"]),
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(json.dumps(materialized_summary, indent=2) + "\n")

    print(
        f"Built dataset at {output_dir} with "
        f"{summary['train_images']} train images, {summary['val_images']} val images, "
        f"and {summary['excluded_images']} excluded duplicates."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("extracted/Fotos_para_entrenamiento"),
        help="Directory containing the original image files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("annotations/manifest.csv"),
        help="CSV manifest with reviewed class labels and bounding boxes.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("annotations/classes.txt"),
        help="Text file with one class name per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory where the YOLO dataset will be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Square output size used for preprocessing and labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        classes_path=args.classes,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
