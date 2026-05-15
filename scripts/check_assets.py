#!/usr/bin/env python3
"""
Preflight asset checker for MNER / GMNER / FMNERG datasets.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_default_dataset, resolve_dataset_split_file
from data.grounding import resolve_vinvl_feature_path, resolve_xml_annotation_path


def collect_image_ids(split_file: str) -> list[str]:
    image_ids: list[str] = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("IMGID:"):
                image_ids.append(line.split("IMGID:", maxsplit=1)[-1].strip())
    return image_ids


def format_examples(items: list[str], limit: int = 5) -> str:
    if not items:
        return "none"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    return ", ".join(shown) + suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Check split / VinVL / XML assets before training.")
    parser.add_argument("--task", type=str, default="mner", choices=["mner", "gmner", "fmnerg"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--vinvl_dir", type=str, default=None)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--splits", type=str, default="train,dev,test")
    parser.add_argument(
        "--strict_xml",
        action="store_true",
        help="Exit non-zero when XML annotations are missing.",
    )
    args = parser.parse_args()

    dataset = args.dataset or get_default_dataset(args.task)
    split_names = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not split_names:
        raise ValueError("No valid splits were provided.")

    print(f"Task: {args.task}")
    print(f"Dataset: {dataset}")
    print(f"Data dir: {args.data_dir}")

    split_files: dict[str, str] = {}
    missing_split_files: list[str] = []
    split_image_ids: dict[str, list[str]] = {}
    all_image_ids: list[str] = []

    for split in split_names:
        try:
            split_file = resolve_dataset_split_file(args.data_dir, dataset, split, args.task)
        except FileNotFoundError as exc:
            missing_split_files.append(str(exc))
            continue

        split_files[split] = split_file
        image_ids = collect_image_ids(split_file)
        split_image_ids[split] = image_ids
        all_image_ids.extend(image_ids)
        print(f"[OK] split={split:<5} file={split_file} samples={len(image_ids)}")

    if missing_split_files:
        print("\n[ERROR] Missing split files:")
        for message in missing_split_files:
            print(message)
        return 1

    if args.task == "mner":
        print("\nMNER uses raw images only; VinVL/XML checks are skipped.")
        return 0

    if not args.vinvl_dir or not args.annotation_dir:
        print("\n[ERROR] GMNER/FMNERG require both --vinvl_dir and --annotation_dir.")
        return 1

    missing_vinvl: list[str] = []
    missing_xml: list[str] = []
    unique_image_ids = sorted(set(all_image_ids))

    for image_id in unique_image_ids:
        try:
            resolve_vinvl_feature_path(args.vinvl_dir, image_id)
        except FileNotFoundError:
            missing_vinvl.append(image_id)

        try:
            resolve_xml_annotation_path(args.annotation_dir, image_id)
        except FileNotFoundError:
            missing_xml.append(image_id)

    print(f"\nUnique image ids: {len(unique_image_ids)}")
    print(f"VinVL dir: {args.vinvl_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Missing VinVL features: {len(missing_vinvl)}")
    print(f"Examples: {format_examples(missing_vinvl)}")
    print(f"Missing XML annotations: {len(missing_xml)}")
    print(f"Examples: {format_examples(missing_xml)}")

    if missing_vinvl:
        return 1
    if missing_xml:
        message = (
            "\nXML annotations are missing for some images. "
            "This is expected when a sample has no grounded entity, but it can also indicate incomplete assets."
        )
        print(message)
        if args.strict_xml:
            return 1
        print("Split files and VinVL features look usable; XML coverage should be checked against your dataset assumptions.")
        return 0

    print("\nAll required split / VinVL / XML assets are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
