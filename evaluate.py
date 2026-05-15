"""
Evaluation for MNER / GMNER / FMNERG.
"""

from __future__ import annotations

import argparse
import os
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from config import (
    FMNERG_FINE_TO_COARSE,
    ImaginEConfig,
    get_default_dataset,
    get_coarse_entity_types,
    get_entity_types,
    get_fine_to_coarse_ids,
    get_id_to_entity_type,
    resolve_dataset_split_file,
)
from data.dataset import TwitterMNERDataset, collate_fn
from data.grounding import compute_iou
from data.processor import MNERProcessor
from models.imagine_model import ImaginEModel
from utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


GROUNDING_DECODE_MODES = ("argmax", "soft_groundable", "hard_groundable")


def _effective_grounding_decode_mode(config: ImaginEConfig) -> str:
    mode = getattr(config.model, "grounding_decode_mode", "argmax")
    if mode not in GROUNDING_DECODE_MODES:
        mode = "argmax"
    if getattr(config.model, "use_groundable_gate", False) and mode == "argmax":
        return "hard_groundable"
    return mode


def _should_cache_groundable_logits(model: ImaginEModel, config: ImaginEConfig) -> bool:
    return (
        getattr(model, "use_groundable_head_for_eval", True)
        and _effective_grounding_decode_mode(config) != "argmax"
    )


def normalize_type_label(label: str) -> str:
    if label == "OTHER":
        return "MISC"
    return label


def compute_span_f1(
    pred_spans: list[set[tuple[int, int, str]]],
    gold_spans: list[set[tuple[int, int, str]]],
    type_names: list[str] | None = None,
) -> dict:
    """Strict span-level P/R/F1 for traditional MNER."""
    tp_total = 0
    fp_total = 0
    fn_total = 0

    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)

    for preds, golds in zip(pred_spans, gold_spans):
        tp = preds & golds
        fp = preds - golds
        fn = golds - preds

        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)

        for span in tp:
            type_tp[span[2]] += 1
        for span in fp:
            type_fp[span[2]] += 1
        for span in fn:
            type_fn[span[2]] += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if type_names is None:
        type_names = ["PER", "LOC", "ORG", "MISC"]

    per_type_f1 = {}
    for type_name in type_names:
        if type_name == "O":
            continue
        tp_t = type_tp[type_name]
        fp_t = type_fp[type_name]
        fn_t = type_fn[type_name]
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1_t = (2 * p_t * r_t / (p_t + r_t)) if (p_t + r_t) > 0 else 0.0
        per_type_f1[type_name] = f1_t

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_type_f1": per_type_f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "task": "mner",
        "main_metric_name": "mner",
    }


def _metric_dict(tp: int, pred_total: int, gold_total: int) -> dict[str, float | int]:
    precision = tp / pred_total if pred_total > 0 else 0.0
    recall = tp / gold_total if gold_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": pred_total - tp,
        "fn": gold_total - tp,
    }


def _greedy_decode_non_overlapping(
    logits_b: torch.Tensor,
    span_indices_b: torch.Tensor,
    span_mask_b: torch.Tensor,
    id_to_entity_type: dict[int, str],
    margin_threshold: float = 0.0,
) -> list[dict[str, object]]:
    """Decode non-overlapping entity spans for one sample."""
    preds = logits_b.argmax(dim=-1)
    scores = logits_b.max(dim=-1).values
    o_logits = logits_b[:, 0]
    entity_max_logits = logits_b[:, 1:].max(dim=-1).values
    margins = entity_max_logits - o_logits

    candidates = []
    for span_idx in range(span_indices_b.size(0)):
        if span_mask_b[span_idx].item() < 0.5:
            continue
        pred_id = int(preds[span_idx].item())
        if pred_id == 0:
            continue
        if margins[span_idx].item() < margin_threshold:
            continue
        start = int(span_indices_b[span_idx, 0].item())
        end = int(span_indices_b[span_idx, 1].item())
        candidates.append({
            "score": float(scores[span_idx].item()),
            "span_idx": span_idx,
            "token_span": (start, end),
            "type_id": pred_id,
            "type_label": id_to_entity_type[pred_id],
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    occupied_positions: set[int] = set()
    for candidate in candidates:
        start, end = candidate["token_span"]
        token_positions = range(start, end + 1)
        if any(pos in occupied_positions for pos in token_positions):
            continue
        occupied_positions.update(token_positions)
        selected.append(candidate)
    return selected


def _apply_fine_rerank(
    logits: torch.Tensor,
    coarse_logits: torch.Tensor | None,
    task: str,
    lambda_value: float,
) -> torch.Tensor:
    """Dev-time FMNERG fine-type rerank with predicted parent coarse scores."""
    if task != "fmnerg" or coarse_logits is None or lambda_value <= 0:
        return logits
    fine_to_coarse = torch.tensor(
        get_fine_to_coarse_ids(task),
        dtype=torch.long,
        device=logits.device,
    )
    if fine_to_coarse.numel() != logits.size(-1):
        return logits
    if int(fine_to_coarse.max().item()) >= coarse_logits.size(-1):
        return logits
    gather_index = fine_to_coarse.view(*([1] * (logits.dim() - 1)), -1).expand(
        *logits.shape[:-1],
        -1,
    )
    parent_scores = F.log_softmax(coarse_logits.float(), dim=-1).gather(
        dim=-1,
        index=gather_index,
    )
    return logits + lambda_value * parent_scores.to(dtype=logits.dtype)


def _token_span_to_word_span(
    token_start: int,
    token_end: int,
    word_bounds: list[tuple[int, int, int]],
) -> tuple[int, int] | None:
    word_start = None
    word_end = None
    for word_idx, first_tok, last_tok in word_bounds:
        if first_tok == token_start:
            word_start = word_idx
        if last_tok == token_end:
            word_end = word_idx
    if word_start is None or word_end is None:
        return None
    return (word_start, word_end)


def _grounding_prediction(
    grounding_logits: torch.Tensor,
    region_boxes: torch.Tensor,
    region_mask: torch.Tensor,
    groundable_logit: torch.Tensor | None = None,
    decode_mode: str = "argmax",
    groundable_threshold: float = 0.0,
    clip_patch_logits: torch.Tensor | None = None,
    clip_patch_boxes: torch.Tensor | None = None,
    clip_patch_mask: torch.Tensor | None = None,
    use_clip_patch_fallback: bool = False,
    clip_fallback_threshold: float = 0.0,
) -> dict[str, object]:
    num_classes = grounding_logits.size(-1)
    valid_regions = int(region_mask.sum().item())
    no_region_idx = num_classes - 1

    def maybe_clip_fallback(base: dict[str, object]) -> dict[str, object]:
        if base["groundable"] or not use_clip_patch_fallback:
            return base
        if groundable_logit is None or float(groundable_logit.item()) <= clip_fallback_threshold:
            return base
        if clip_patch_logits is None or clip_patch_boxes is None:
            return base
        common_patches = min(clip_patch_logits.size(-1), clip_patch_boxes.size(0))
        if clip_patch_mask is not None:
            common_patches = min(common_patches, clip_patch_mask.size(0))
        if common_patches <= 0:
            return base

        patch_logits = clip_patch_logits[:common_patches]
        patch_boxes = clip_patch_boxes[:common_patches]
        if clip_patch_mask is None:
            patch_mask_bool = torch.ones(
                common_patches,
                dtype=torch.bool,
                device=patch_logits.device,
            )
        else:
            patch_mask_bool = clip_patch_mask[:common_patches].to(device=patch_logits.device) >= 0.5
        if int(patch_mask_bool.sum().item()) <= 0:
            return base

        adjusted = patch_logits.clone().float()
        adjusted = adjusted.masked_fill(~patch_mask_bool, -1e4)
        pred_idx = int(adjusted.argmax(dim=-1).item())
        if pred_idx >= patch_boxes.size(0) or not bool(patch_mask_bool[pred_idx].item()):
            return base
        return {
            "groundable": True,
            "region_index": f"clip_patch:{pred_idx}",
            "box": patch_boxes[pred_idx].tolist(),
        }

    if decode_mode == "hard_groundable" and groundable_logit is not None:
        is_groundable = float(groundable_logit.item()) > groundable_threshold
        if not is_groundable or valid_regions <= 0:
            return maybe_clip_fallback({
                "groundable": False,
                "region_index": None,
                "box": None,
            })
        pred_idx = int(grounding_logits[:valid_regions].argmax(dim=-1).item())
        return {
            "groundable": True,
            "region_index": pred_idx,
            "box": region_boxes[pred_idx].tolist(),
        }

    if decode_mode == "soft_groundable" and groundable_logit is not None:
        if valid_regions <= 0:
            return maybe_clip_fallback({
                "groundable": False,
                "region_index": None,
                "box": None,
            })

        adjusted = grounding_logits.clone().float()
        if valid_regions < no_region_idx:
            adjusted[valid_regions:no_region_idx] = -1e4

        groundable_prob = torch.sigmoid(groundable_logit.float() - groundable_threshold)
        adjusted[:valid_regions] = adjusted[:valid_regions] + torch.log(
            groundable_prob.clamp(min=1e-6)
        )
        adjusted[no_region_idx] = adjusted[no_region_idx] + torch.log(
            (1.0 - groundable_prob).clamp(min=1e-6)
        )
        pred_idx = int(adjusted.argmax(dim=-1).item())
        if pred_idx == no_region_idx or pred_idx >= valid_regions:
            return maybe_clip_fallback({
                "groundable": False,
                "region_index": None,
                "box": None,
            })
        return {
            "groundable": True,
            "region_index": pred_idx,
            "box": region_boxes[pred_idx].tolist(),
        }

    pred_idx = int(grounding_logits.argmax(dim=-1).item())
    if pred_idx == no_region_idx or pred_idx >= valid_regions:
        return maybe_clip_fallback({
            "groundable": False,
            "region_index": None,
            "box": None,
        })

    return {
        "groundable": True,
        "region_index": pred_idx,
        "box": region_boxes[pred_idx].tolist(),
    }


def _build_pred_triplets(
    type_logits: torch.Tensor,
    grounding_logits: torch.Tensor,
    groundable_logits: torch.Tensor | None,
    span_indices: torch.Tensor,
    span_mask: torch.Tensor,
    region_boxes: torch.Tensor,
    region_mask: torch.Tensor,
    clip_patch_logits: torch.Tensor | None,
    clip_patch_boxes: torch.Tensor | None,
    clip_patch_mask: torch.Tensor | None,
    metadata: dict,
    id_to_entity_type: dict[int, str],
    task: str,
    margin_threshold: float,
    grounding_decode_mode: str,
    groundable_threshold: float,
    use_clip_patch_fallback: bool = False,
    clip_fallback_threshold: float = 0.0,
) -> list[dict[str, object]]:
    decoded_spans = _greedy_decode_non_overlapping(
        type_logits,
        span_indices,
        span_mask,
        id_to_entity_type=id_to_entity_type,
        margin_threshold=margin_threshold,
    )
    triplets = []
    for candidate in decoded_spans:
        word_span = _token_span_to_word_span(
            candidate["token_span"][0],
            candidate["token_span"][1],
            metadata["word_bounds"],
        )
        if word_span is None:
            continue
        word_start, word_end = word_span
        entity_text = " ".join(metadata["words"][word_start:word_end + 1])
        grounding = _grounding_prediction(
            grounding_logits[candidate["span_idx"]],
            region_boxes,
            region_mask,
            (
                groundable_logits[candidate["span_idx"]]
                if groundable_logits is not None
                else None
            ),
            decode_mode=grounding_decode_mode,
            groundable_threshold=groundable_threshold,
            clip_patch_logits=(
                clip_patch_logits[candidate["span_idx"]]
                if clip_patch_logits is not None
                else None
            ),
            clip_patch_boxes=clip_patch_boxes,
            clip_patch_mask=clip_patch_mask,
            use_clip_patch_fallback=use_clip_patch_fallback,
            clip_fallback_threshold=clip_fallback_threshold,
        )
        type_label = normalize_type_label(candidate["type_label"])
        triplets.append({
            "entity": entity_text,
            "type": type_label,
            "coarse_type": (
                FMNERG_FINE_TO_COARSE.get(type_label)
                if task == "fmnerg"
                else type_label
            ),
            "word_span": word_span,
            "groundable": grounding["groundable"],
            "box": grounding["box"],
            "region_index": grounding["region_index"],
        })
    return triplets


def _build_gold_triplets(metadata: dict, task: str) -> list[dict[str, object]]:
    triplets = []
    for gold in metadata["gold_entities"]:
        type_label = normalize_type_label(str(gold["type_label"]))
        triplets.append({
            "entity": str(gold["entity_text"]),
            "type": type_label,
            "coarse_type": (
                FMNERG_FINE_TO_COARSE.get(type_label)
                if task == "fmnerg"
                else normalize_type_label(str(gold["coarse_type"]))
            ),
            "word_span": tuple(gold["word_span"]),
            "groundable": bool(gold["groundable"]),
            "detector_miss": bool(gold["detector_miss"]),
            "gt_boxes": [list(box) for box in gold["gt_boxes"]],
        })
    return triplets


def _triplets_to_dict(
    triplets: list[dict[str, object]],
    key_builder,
) -> dict[tuple, dict[str, object]]:
    result = {}
    for triplet in triplets:
        key = key_builder(triplet)
        if key is None:
            continue
        result[key] = triplet
    return result


def _grounding_match(pred: dict[str, object], gold: dict[str, object]) -> bool:
    if not gold["groundable"]:
        return pred.get("box") is None
    pred_box = pred.get("box")
    if pred_box is None:
        return False
    return any(compute_iou(pred_box, gt_box) > 0.5 for gt_box in gold.get("gt_boxes", []))


def _structured_metric(
    pred_samples: list[list[dict[str, object]]],
    gold_samples: list[list[dict[str, object]]],
    key_builder,
    matcher,
) -> dict[str, float | int]:
    tp = 0
    pred_total = 0
    gold_total = 0

    for pred_triplets, gold_triplets in zip(pred_samples, gold_samples):
        pred_dict = _triplets_to_dict(pred_triplets, key_builder)
        gold_dict = _triplets_to_dict(gold_triplets, key_builder)

        pred_total += len(pred_dict)
        gold_total += len(gold_dict)

        for key, pred_triplet in pred_dict.items():
            gold_triplet = gold_dict.get(key)
            if gold_triplet is not None and matcher(pred_triplet, gold_triplet):
                tp += 1

    return _metric_dict(tp, pred_total, gold_total)


def _structured_metric_scan_gold(
    pred_samples: list[list[dict[str, object]]],
    gold_samples: list[list[dict[str, object]]],
    pred_key_builder,
    gold_key_builder,
    matcher,
) -> dict[str, float | int]:
    """Match each deduplicated prediction against scanned gold values."""
    tp = 0
    pred_total = 0
    gold_total = 0

    for pred_triplets, gold_triplets in zip(pred_samples, gold_samples):
        pred_dict = _triplets_to_dict(pred_triplets, pred_key_builder)
        gold_dict = _triplets_to_dict(gold_triplets, gold_key_builder)

        pred_total += len(pred_dict)
        gold_total += len(gold_dict)

        for pred_triplet in pred_dict.values():
            matched = False
            for gold_triplet in gold_dict.values():
                if matcher(pred_triplet, gold_triplet):
                    matched = True
                    break
            if matched:
                tp += 1

    return _metric_dict(tp, pred_total, gold_total)


def compute_grounded_metrics(
    pred_samples: list[list[dict[str, object]]],
    gold_samples: list[list[dict[str, object]]],
    task: str,
) -> dict:
    """Compute official-style metrics for GMNER / FMNERG."""
    if task == "gmner":
        overall = _structured_metric(
            pred_samples,
            gold_samples,
            key_builder=lambda triplet: triplet["word_span"],
            matcher=lambda pred, gold: (
                pred["type"] == gold["type"] and _grounding_match(pred, gold)
            ),
        )
        text_only = _structured_metric(
            pred_samples,
            gold_samples,
            key_builder=lambda triplet: triplet["word_span"],
            matcher=lambda pred, gold: pred["type"] == gold["type"],
        )
        eeg = _structured_metric(
            pred_samples,
            gold_samples,
            key_builder=lambda triplet: triplet["word_span"],
            matcher=_grounding_match,
        )
        return {
            **overall,
            "task": "gmner",
            "main_metric_name": "gmner",
            "subtasks": {
                "mner": text_only,
                "eeg": eeg,
            },
        }

    overall = _structured_metric(
        pred_samples,
        gold_samples,
        key_builder=lambda triplet: (
            triplet["entity"],
            triplet["type"],
            triplet["groundable"],
        ),
        matcher=_grounding_match,
    )
    text_only = _structured_metric(
        pred_samples,
        gold_samples,
        key_builder=lambda triplet: (triplet["entity"], triplet["type"]),
        matcher=lambda pred, gold: True,
    )
    eeg = _structured_metric_scan_gold(
        pred_samples,
        gold_samples,
        pred_key_builder=lambda triplet: (
            triplet["entity"],
            triplet["type"],
            triplet["groundable"],
        ),
        gold_key_builder=lambda triplet: (
            triplet["entity"],
            triplet["type"],
            triplet["groundable"],
        ),
        matcher=lambda pred, gold: (
            pred["entity"] == gold["entity"]
            and pred["groundable"] == gold["groundable"]
            and _grounding_match(pred, gold)
        ),
    )
    return {
        **overall,
        "task": "fmnerg",
        "main_metric_name": "fmnerg",
        "subtasks": {
            "fmner": text_only,
            "eeg": eeg,
        },
    }


def format_metrics_for_logging(metrics: dict) -> list[str]:
    """Convert a metrics dict into concise log lines."""
    lines = [
        f"{metrics['main_metric_name'].upper()} | "
        f"P: {metrics['precision']:.4f} "
        f"R: {metrics['recall']:.4f} "
        f"F1: {metrics['f1']:.4f}"
    ]
    if "subtasks" in metrics:
        for subtask_name, subtask_metrics in metrics["subtasks"].items():
            lines.append(
                f"  {subtask_name.upper()} | "
                f"P: {subtask_metrics['precision']:.4f} "
                f"R: {subtask_metrics['recall']:.4f} "
                f"F1: {subtask_metrics['f1']:.4f}"
            )
    elif "per_type_f1" in metrics:
        for type_name, value in metrics["per_type_f1"].items():
            lines.append(f"  {type_name}: F1={value:.4f}")
    return lines


def _prepare_model_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    model_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": batch["pixel_values"].to(device),
        "region_features": batch["region_features"].to(device),
        "region_mask": batch["region_mask"].to(device),
        "span_indices": batch["span_indices"].to(device),
    }
    if "knowledge_input_ids" in batch and "knowledge_attention_mask" in batch:
        model_inputs["knowledge_input_ids"] = batch["knowledge_input_ids"].to(device)
        model_inputs["knowledge_attention_mask"] = batch["knowledge_attention_mask"].to(device)
    return model_inputs


def _evaluate_cached_batches(
    cached_batches: list[dict],
    config: ImaginEConfig,
    margin_threshold: float,
    groundable_threshold: float | None = None,
    fine_rerank_lambda: float | None = None,
) -> dict:
    task = config.train.task
    id_to_entity_type = get_id_to_entity_type(task)
    grounding_decode_mode = _effective_grounding_decode_mode(config)
    if groundable_threshold is None:
        groundable_threshold = getattr(config.model, "groundable_threshold", 0.0)
    if fine_rerank_lambda is None:
        fine_rerank_lambda = getattr(config.model, "fine_rerank_lambda", 0.0)

    if task == "mner":
        all_pred_spans = []
        all_gold_spans = []
        for cached in cached_batches:
            logits = _apply_fine_rerank(
                cached["logits"],
                cached.get("coarse_logits"),
                task,
                fine_rerank_lambda,
            )
            span_indices = cached["span_indices"]
            span_labels = cached["span_labels"]
            span_mask = cached["span_mask"]
            metadata = cached["metadata"]
            batch_size = logits.size(0)
            for batch_idx in range(batch_size):
                gold_triplets = _build_gold_triplets(metadata[batch_idx], task=task)
                pred_set = set()
                gold_set = set()
                decoded_spans = _greedy_decode_non_overlapping(
                    logits[batch_idx],
                    span_indices[batch_idx],
                    span_mask[batch_idx],
                    id_to_entity_type=id_to_entity_type,
                    margin_threshold=margin_threshold,
                )
                for candidate in decoded_spans:
                    word_span = _token_span_to_word_span(
                        candidate["token_span"][0],
                        candidate["token_span"][1],
                        metadata[batch_idx]["word_bounds"],
                    )
                    if word_span is not None:
                        pred_set.add((
                            word_span[0],
                            word_span[1],
                            normalize_type_label(candidate["type_label"]),
                        ))
                for gold in gold_triplets:
                    gold_set.add((
                        gold["word_span"][0],
                        gold["word_span"][1],
                        gold["type"],
                    ))
                all_pred_spans.append(pred_set)
                all_gold_spans.append(gold_set)
        return compute_span_f1(all_pred_spans, all_gold_spans, type_names=get_entity_types(task))

    pred_samples = []
    gold_samples = []
    for cached in cached_batches:
        logits = _apply_fine_rerank(
            cached["logits"],
            cached.get("coarse_logits"),
            task,
            fine_rerank_lambda,
        )
        batch_size = logits.size(0)
        for batch_idx in range(batch_size):
            pred_samples.append(_build_pred_triplets(
                logits[batch_idx],
                cached["grounding_logits"][batch_idx],
                (
                    cached["groundable_logits"][batch_idx]
                    if cached.get("groundable_logits") is not None
                    else None
                ),
                cached["span_indices"][batch_idx],
                cached["span_mask"][batch_idx],
                cached["region_boxes"][batch_idx],
                cached["region_mask"][batch_idx],
                (
                    cached["clip_patch_logits"][batch_idx]
                    if cached.get("clip_patch_logits") is not None
                    else None
                ),
                cached.get("clip_patch_boxes", None)[batch_idx]
                if cached.get("clip_patch_boxes", None) is not None
                else None,
                cached.get("clip_patch_mask", None)[batch_idx]
                if cached.get("clip_patch_mask", None) is not None
                else None,
                cached["metadata"][batch_idx],
                id_to_entity_type=id_to_entity_type,
                task=task,
                margin_threshold=margin_threshold,
                grounding_decode_mode=grounding_decode_mode,
                groundable_threshold=groundable_threshold,
                use_clip_patch_fallback=getattr(config.model, "use_clip_patch_fallback", False),
                clip_fallback_threshold=getattr(config.model, "clip_fallback_threshold", 0.0),
            ))
            gold_samples.append(_build_gold_triplets(cached["metadata"][batch_idx], task=task))
    return compute_grounded_metrics(pred_samples, gold_samples, task=task)


@torch.no_grad()
def evaluate_model(
    model: ImaginEModel,
    dataloader: DataLoader,
    config: ImaginEConfig,
    device: torch.device,
    margin_threshold: float = 0.0,
    groundable_threshold: float | None = None,
    fine_rerank_lambda: float | None = None,
) -> dict:
    model.eval()
    cached_batches = []
    use_groundable_head = _should_cache_groundable_logits(model, config)
    for batch in dataloader:
        model_inputs = _prepare_model_inputs(batch, device)
        with autocast("cuda", enabled=config.train.fp16):
            outputs = model(**model_inputs)
        cached_batches.append({
            "logits": outputs["logits"].cpu(),
            "coarse_logits": (
                outputs["coarse_logits"].cpu()
                if outputs.get("coarse_logits") is not None
                else None
            ),
            "grounding_logits": outputs["grounding_logits"].cpu(),
            "groundable_logits": (
                outputs["groundable_logits"].cpu()
                if use_groundable_head and "groundable_logits" in outputs
                else None
            ),
            "clip_patch_logits": (
                outputs["clip_patch_logits"].cpu()
                if outputs.get("clip_patch_logits") is not None
                else None
            ),
            "span_indices": batch["span_indices"].cpu(),
            "span_labels": batch["span_labels"].cpu(),
            "span_mask": batch["span_mask"].cpu(),
            "region_boxes": batch["region_boxes"].cpu(),
            "region_mask": batch["region_mask"].cpu(),
            "clip_patch_boxes": batch["clip_patch_boxes"].cpu(),
            "clip_patch_mask": batch["clip_patch_mask"].cpu(),
            "metadata": batch["metadata"],
        })
    return _evaluate_cached_batches(
        cached_batches,
        config,
        margin_threshold=margin_threshold,
        groundable_threshold=groundable_threshold,
        fine_rerank_lambda=fine_rerank_lambda,
    )


@torch.no_grad()
def tune_threshold_on_dev(
    model: ImaginEModel,
    dataloader: DataLoader,
    config: ImaginEConfig,
    device: torch.device,
    thresholds: list[float] | None = None,
    groundable_thresholds: list[float] | None = None,
) -> tuple[float, float, float, dict]:
    """Tune entity margin and, optionally, groundable decode thresholds."""
    if thresholds is None:
        thresholds = [i * 0.1 for i in range(21)]
    if groundable_thresholds is None:
        groundable_thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    model.eval()
    cached_batches = []
    use_groundable_head = _should_cache_groundable_logits(model, config)
    for batch in dataloader:
        model_inputs = _prepare_model_inputs(batch, device)
        with autocast("cuda", enabled=config.train.fp16):
            outputs = model(**model_inputs)
        cached_batches.append({
            "logits": outputs["logits"].cpu(),
            "coarse_logits": (
                outputs["coarse_logits"].cpu()
                if outputs.get("coarse_logits") is not None
                else None
            ),
            "grounding_logits": outputs["grounding_logits"].cpu(),
            "groundable_logits": (
                outputs["groundable_logits"].cpu()
                if use_groundable_head and "groundable_logits" in outputs
                else None
            ),
            "clip_patch_logits": (
                outputs["clip_patch_logits"].cpu()
                if outputs.get("clip_patch_logits") is not None
                else None
            ),
            "span_indices": batch["span_indices"].cpu(),
            "span_labels": batch["span_labels"].cpu(),
            "span_mask": batch["span_mask"].cpu(),
            "region_boxes": batch["region_boxes"].cpu(),
            "region_mask": batch["region_mask"].cpu(),
            "clip_patch_boxes": batch["clip_patch_boxes"].cpu(),
            "clip_patch_mask": batch["clip_patch_mask"].cpu(),
            "metadata": batch["metadata"],
        })

    best_threshold = 0.0
    best_groundable_threshold = getattr(config.model, "groundable_threshold", 0.0)
    best_fine_rerank_lambda = getattr(config.model, "fine_rerank_lambda", 0.0)
    best_metrics = None
    best_f1 = -1.0
    best_eeg_f1 = -1.0
    grounding_decode_mode = _effective_grounding_decode_mode(config)
    thresholds_to_try = (
        groundable_thresholds
        if getattr(config.train, "tune_grounding_decode", False)
        and grounding_decode_mode != "argmax"
        else [best_groundable_threshold]
    )
    fine_lambdas_to_try = (
        getattr(config.train, "fine_rerank_lambdas", [0.0])
        if getattr(config.train, "use_fine_rerank", False)
        and config.train.task == "fmnerg"
        else [best_fine_rerank_lambda]
    )
    for threshold in thresholds:
        for g_threshold in thresholds_to_try:
            for fine_lambda in fine_lambdas_to_try:
                metrics = _evaluate_cached_batches(
                    cached_batches,
                    config,
                    margin_threshold=threshold,
                    groundable_threshold=g_threshold,
                    fine_rerank_lambda=fine_lambda,
                )
                eeg_f1 = metrics.get("subtasks", {}).get("eeg", {}).get("f1", 0.0)
                if (
                    metrics["f1"] > best_f1 + 1e-12
                    or (
                        abs(metrics["f1"] - best_f1) <= 1e-12
                        and eeg_f1 > best_eeg_f1
                    )
                ):
                    best_f1 = metrics["f1"]
                    best_eeg_f1 = eeg_f1
                    best_threshold = threshold
                    best_groundable_threshold = g_threshold
                    best_fine_rerank_lambda = fine_lambda
                    best_metrics = metrics

    assert best_metrics is not None
    logger.info(
        "Threshold search: margin=%.2f groundable=%.2f fine_lambda=%.2f -> P=%.4f R=%.4f F1=%.4f EEG_F1=%.4f",
        best_threshold,
        best_groundable_threshold,
        best_fine_rerank_lambda,
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1"],
        best_eeg_f1,
    )
    return best_threshold, best_groundable_threshold, best_fine_rerank_lambda, best_metrics


def _load_config_from_checkpoint(
    args,
    device: torch.device,
) -> tuple[ImaginEConfig, dict]:
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ImaginEConfig()

    if "model_config" in ckpt:
        for key, value in ckpt["model_config"].items():
            setattr(config.model, key, value)
    if "train_config" in ckpt:
        for key, value in ckpt["train_config"].items():
            setattr(config.train, key, value)
    if "loss_config" in ckpt:
        for key, value in ckpt["loss_config"].items():
            setattr(config.loss, key, value)

    if args.task is not None:
        config.train.task = args.task
        config.model.task = args.task
    if args.dataset is not None:
        config.train.dataset = args.dataset
    elif "train_config" not in ckpt:
        config.train.dataset = get_default_dataset(config.train.task)
    if args.visual_backend is not None:
        config.model.visual_backend = args.visual_backend
    if args.data_dir is not None:
        config.train.data_dir = args.data_dir
    if args.image_dir is not None:
        config.train.image_dir = args.image_dir
    if args.vinvl_dir is not None:
        config.train.vinvl_dir = args.vinvl_dir
    if args.annotation_dir is not None:
        config.train.annotation_dir = args.annotation_dir
    if args.text_model is not None:
        config.model.text_model_name = args.text_model
    if args.image_model is not None:
        config.model.image_model_name = args.image_model
    if args.shared_dim is not None:
        config.model.shared_dim = args.shared_dim
    if args.max_regions is not None:
        config.model.max_regions = args.max_regions
    if args.vinvl_feature_dim is not None:
        config.model.vinvl_feature_dim = args.vinvl_feature_dim
    if args.use_groundable_gate:
        config.model.use_groundable_gate = True
    if args.grounding_decode_mode is not None:
        config.model.grounding_decode_mode = args.grounding_decode_mode
    if args.groundable_threshold is not None:
        config.model.groundable_threshold = args.groundable_threshold
    if getattr(args, "append_caption", False):
        config.train.append_caption = True
    if getattr(args, "caption_files", None) is not None:
        config.train.caption_files = args.caption_files
    if getattr(args, "caption_max_words", None) is not None:
        config.train.caption_max_words = args.caption_max_words
    if getattr(args, "use_clip_patch_fallback", False):
        config.model.use_clip_patch_fallback = True
    if getattr(args, "clip_fallback_threshold", None) is not None:
        config.model.clip_fallback_threshold = args.clip_fallback_threshold
    if getattr(args, "knowledge_injection", None) is not None:
        config.model.knowledge_injection = args.knowledge_injection
    if getattr(args, "knowledge_files", None) is not None:
        config.train.knowledge_files = args.knowledge_files
    if getattr(args, "knowledge_max_words", None) is not None:
        config.train.knowledge_max_words = args.knowledge_max_words
    if getattr(args, "knowledge_dropout", None) is not None:
        config.model.knowledge_dropout = args.knowledge_dropout
        config.train.knowledge_dropout = args.knowledge_dropout
    if getattr(args, "knowledge_gate_init", None) is not None:
        config.model.knowledge_gate_init = args.knowledge_gate_init
    if getattr(args, "use_type_aware_region_pointer", False):
        config.model.use_type_aware_region_pointer = True
    if getattr(args, "use_fine_rerank", False):
        config.train.use_fine_rerank = True
    if getattr(args, "fine_rerank_lambda", None) is not None:
        config.model.fine_rerank_lambda = args.fine_rerank_lambda

    config.train.fp16 = False
    config.train.device = str(device)
    config.model.num_types = len(get_entity_types(config.train.task))
    config.model.num_coarse_types = len(get_coarse_entity_types(config.train.task))
    return config, ckpt


def main():
    parser = argparse.ArgumentParser(description="Evaluate ImaginE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--vinvl_dir", type=str, default=None)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--visual_backend", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_model", type=str, default=None)
    parser.add_argument("--image_model", type=str, default=None)
    parser.add_argument("--shared_dim", type=int, default=None)
    parser.add_argument("--max_regions", type=int, default=None)
    parser.add_argument("--vinvl_feature_dim", type=int, default=None)
    parser.add_argument("--use_groundable_gate", action="store_true")
    parser.add_argument(
        "--grounding_decode_mode",
        type=str,
        default=None,
        choices=GROUNDING_DECODE_MODES,
    )
    parser.add_argument("--groundable_threshold", type=float, default=None)
    parser.add_argument("--append_caption", action="store_true")
    parser.add_argument("--caption_files", nargs="*", default=None)
    parser.add_argument("--caption_max_words", type=int, default=None)
    parser.add_argument("--use_clip_patch_fallback", action="store_true")
    parser.add_argument("--clip_fallback_threshold", type=float, default=None)
    parser.add_argument(
        "--knowledge_injection",
        type=str,
        default=None,
        choices=["off", "gated_span"],
    )
    parser.add_argument("--knowledge_files", nargs="*", default=None)
    parser.add_argument("--knowledge_max_words", type=int, default=None)
    parser.add_argument("--knowledge_dropout", type=float, default=None)
    parser.add_argument("--knowledge_gate_init", type=float, default=None)
    parser.add_argument("--use_type_aware_region_pointer", action="store_true")
    parser.add_argument("--use_fine_rerank", action="store_true")
    parser.add_argument("--fine_rerank_lambda", type=float, default=None)
    args = parser.parse_args()

    set_seed(42)
    setup_logging()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu" if args.device.startswith("cuda") else args.device)

    config, ckpt = _load_config_from_checkpoint(args, device)
    if not config.train.dataset:
        config.train.dataset = get_default_dataset(config.train.task)
    config.train.fp16 = config.train.fp16 and device.type == "cuda"

    if config.train.task in {"gmner", "fmnerg"} and config.model.visual_backend != "vinvl":
        raise ValueError("GMNER/FMNERG evaluation currently requires --visual_backend vinvl.")
    if config.model.visual_backend == "vinvl" and (
        not config.train.vinvl_dir or not config.train.annotation_dir
    ):
        raise ValueError("VinVL evaluation requires both --vinvl_dir and --annotation_dir.")

    data_file = resolve_dataset_split_file(
        data_dir=config.train.data_dir,
        dataset=config.train.dataset,
        split=args.split,
        task=config.train.task,
    )

    processor = MNERProcessor(
        text_model_name=config.model.text_model_name,
        image_model_name=config.model.image_model_name,
        max_seq_length=config.model.max_seq_length,
        max_span_length=config.model.max_span_length,
        visual_backend=config.model.visual_backend,
        load_images_for_vinvl=config.model.use_clip_patch_fallback,
    )

    dataset = TwitterMNERDataset(
        data_file=data_file,
        image_dir=config.train.image_dir,
        processor=processor,
        is_train=False,
        task=config.train.task,
        visual_backend=config.model.visual_backend,
        vinvl_dir=config.train.vinvl_dir,
        annotation_dir=config.train.annotation_dir,
        max_regions=config.model.max_regions,
        vinvl_feature_dim=config.model.vinvl_feature_dim,
        normalize_vinvl=config.train.normalize_vinvl,
        append_caption=config.train.append_caption,
        caption_files=config.train.caption_files,
        caption_max_words=config.train.caption_max_words,
        knowledge_files=config.train.knowledge_files,
        knowledge_max_words=config.train.knowledge_max_words,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = ImaginEModel(config.model).to(device)
    state_dict = ckpt["ema_state_dict"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
    has_groundable_head = any("groundable_head" in key for key in state_dict)
    model.use_groundable_head_for_eval = has_groundable_head
    if "ema_state_dict" in ckpt:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded EMA weights from %s", args.checkpoint)
    else:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint from %s", args.checkpoint)
    if not has_groundable_head:
        logger.info("Checkpoint has no groundable head; falling back to grounding argmax.")

    logger.info(
        "Evaluating task=%s dataset=%s split=%s (%d samples)",
        config.train.task,
        config.train.dataset,
        args.split,
        len(dataset),
    )
    metrics = evaluate_model(model, dataloader, config, device)
    for line in format_metrics_for_logging(metrics):
        logger.info(line)


if __name__ == "__main__":
    main()
