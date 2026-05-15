"""
Torch-free official-style metrics for MNER / GMNER / FMNERG.
"""

from __future__ import annotations

from collections import defaultdict

from data.grounding import compute_iou


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
