#!/usr/bin/env python
"""Diagnose VinVL proposal coverage for grounded entities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_default_dataset, resolve_dataset_split_file
from data.dataset import TwitterMNERDataset
from data.grounding import (
    annotation_boxes_by_name,
    build_aspect_iou_map,
    build_grounding_supervision,
    load_vinvl_features,
    resolve_xml_annotation_path,
)
from data.span_utils import extract_entity_spans


def _empty_stats() -> dict[str, int | float]:
    return {
        "total_entities": 0,
        "groundable_entities": 0,
        "covered_groundable": 0,
        "detector_miss": 0,
        "ungroundable_entities": 0,
        "missing_features": 0,
        "missing_annotations": 0,
        "coverage": 0.0,
    }


def diagnose_split(args: argparse.Namespace, split: str) -> dict[str, int | float]:
    split_file = resolve_dataset_split_file(
        args.data_dir,
        args.dataset,
        split,
        args.task,
    )
    dataset = TwitterMNERDataset(
        data_file=split_file,
        image_dir=args.image_dir or "",
        processor=None,
        max_spans=1,
        is_train=False,
        task=args.task,
        visual_backend="vinvl",
        vinvl_dir=args.vinvl_dir,
        annotation_dir=args.annotation_dir,
        max_regions=args.max_regions,
        vinvl_feature_dim=args.vinvl_feature_dim,
        normalize_vinvl=args.normalize_vinvl,
    )
    stats = _empty_stats()

    for sample in dataset.samples:
        entity_spans = extract_entity_spans(sample["labels"])
        if not entity_spans:
            continue

        image_id = sample["image_id"]
        annotation_box_map: dict[str, list[list[int]]] = {}
        try:
            xml_path = resolve_xml_annotation_path(args.annotation_dir, image_id)
            annotation_box_map = annotation_boxes_by_name(xml_path)
        except (FileNotFoundError, TypeError):
            stats["missing_annotations"] += len(entity_spans)

        feature_available = True
        aspect_iou_map = {}
        try:
            _features, proposal_boxes, _region_mask = load_vinvl_features(
                args.vinvl_dir,
                image_id,
                max_regions=args.max_regions,
                feature_dim=args.vinvl_feature_dim,
                normalize=args.normalize_vinvl,
            )
            if annotation_box_map:
                aspect_iou_map = build_aspect_iou_map(annotation_box_map, proposal_boxes)
        except (FileNotFoundError, KeyError, ValueError):
            feature_available = False
            stats["missing_features"] += len(entity_spans)

        for word_start, word_end, _entity_type in entity_spans:
            stats["total_entities"] += 1
            entity_text = " ".join(sample["words"][word_start:word_end + 1])
            if not annotation_box_map.get(entity_text):
                stats["ungroundable_entities"] += 1
                continue

            stats["groundable_entities"] += 1
            if not feature_available:
                stats["detector_miss"] += 1
                continue

            supervision = build_grounding_supervision(
                entity_name=entity_text,
                annotation_box_map=annotation_box_map,
                aspect_iou_map=aspect_iou_map,
                max_regions=args.max_regions,
            )
            if supervision["detector_miss"]:
                stats["detector_miss"] += 1
                continue

            distribution = np.asarray(supervision["distribution"], dtype=np.float32)
            if float(distribution[:-1].sum()) > 0:
                stats["covered_groundable"] += 1
            else:
                stats["detector_miss"] += 1

    groundable = int(stats["groundable_entities"])
    if groundable > 0:
        stats["coverage"] = float(stats["covered_groundable"]) / groundable
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose VinVL top-K entity coverage")
    parser.add_argument("--task", choices=["gmner", "fmnerg"], required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--vinvl_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max_regions", type=int, default=36)
    parser.add_argument("--vinvl_feature_dim", type=int, default=2048)
    parser.add_argument("--normalize_vinvl", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    args.dataset = args.dataset or get_default_dataset(args.task)

    result = {
        "task": args.task,
        "dataset": args.dataset,
        "max_regions": args.max_regions,
        "splits": {
            split: diagnose_split(args, split)
            for split in args.splits
        },
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output_json:
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
