"""
Evaluation script for ImaginE.

Computes strict-match span-level NER metrics:
    - Overall Precision, Recall, F1
    - Per-type (PER, LOC, ORG, MISC) F1
    - Entity boundary AND type must both be correct for a true positive
"""

import argparse
import os
import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import ImaginEConfig, ENTITY_TYPES, ID_TO_ENTITY_TYPE
from models.imagine_model import ImaginEModel
from data.dataset import TwitterMNERDataset, collate_fn
from data.processor import MNERProcessor
from utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def compute_span_f1(
    pred_spans: list[set[tuple[int, int, str]]],
    gold_spans: list[set[tuple[int, int, str]]],
) -> dict:
    """Compute strict-match span-level P/R/F1.

    Each span is a tuple (start, end, type_str). A prediction is correct
    only if both the boundary (start, end) and entity type match exactly.

    Args:
        pred_spans: list (one per sample) of sets of predicted spans
        gold_spans: list (one per sample) of sets of gold spans
    Returns:
        dict with precision, recall, f1, per_type_f1
    """
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

    per_type_f1 = {}
    for etype in ["PER", "LOC", "ORG", "MISC"]:
        tp_t = type_tp[etype]
        fp_t = type_fp[etype]
        fn_t = type_fn[etype]
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1_t = (2 * p_t * r_t / (p_t + r_t)) if (p_t + r_t) > 0 else 0.0
        per_type_f1[etype] = f1_t

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_type_f1": per_type_f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def _greedy_decode_non_overlapping(
    logits_b: torch.Tensor,
    span_indices_b: torch.Tensor,
    span_mask_b: torch.Tensor,
    margin_threshold: float = 0.0,
) -> set[tuple[int, int, str]]:
    """Greedy non-overlapping span decoding for a single sample.

    Selects entity spans by descending confidence, skipping any that
    overlap with an already-selected span. Optionally filters by the
    logit margin between the best entity class and the O class.
    """
    preds = logits_b.argmax(dim=-1)       # (S,)
    scores = logits_b.max(dim=-1).values  # (S,)

    o_logits = logits_b[:, 0]                          # (S,)
    entity_max_logits = logits_b[:, 1:].max(dim=-1).values  # (S,)
    margins = entity_max_logits - o_logits              # (S,)

    candidates = []
    for s_idx in range(span_indices_b.size(0)):
        if span_mask_b[s_idx].item() < 0.5:
            continue
        pred_id = preds[s_idx].item()
        if pred_id == 0:
            continue
        pred_type = ID_TO_ENTITY_TYPE.get(pred_id, "O")
        if pred_type == "O":
            continue
        if margins[s_idx].item() < margin_threshold:
            continue
        start = span_indices_b[s_idx, 0].item()
        end = span_indices_b[s_idx, 1].item()
        candidates.append((scores[s_idx].item(), start, end, pred_type))

    candidates.sort(key=lambda x: x[0], reverse=True)

    pred_set = set()
    occupied = set()
    for _score, start, end, type_str in candidates:
        span_positions = range(start, end + 1)
        if any(pos in occupied for pos in span_positions):
            continue
        occupied.update(span_positions)
        pred_set.add((start, end, type_str))
    return pred_set


@torch.no_grad()
def evaluate_model(
    model: ImaginEModel,
    dataloader: DataLoader,
    config: ImaginEConfig,
    device: torch.device,
    margin_threshold: float = 0.0,
) -> dict:
    """Run evaluation on a dataset.

    Uses greedy non-overlapping span decoding: for each sample, candidate
    spans are ranked by max logit score and greedily selected so that no
    two predicted entities overlap in token positions.

    Args:
        model: the ImaginE model
        dataloader: evaluation data loader
        config: configuration
        device: torch device
        margin_threshold: minimum logit margin (entity - O) to accept a span
    Returns:
        dict with precision, recall, f1, per_type_f1
    """
    model.eval()

    all_pred_spans = []
    all_gold_spans = []

    total_pred_entities = 0
    total_gold_entities = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast("cuda", enabled=config.train.fp16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                span_indices=batch["span_indices"],
            )

        logits = outputs["logits"]  # (B, S, K)

        B = logits.size(0)
        span_indices = batch["span_indices"]  # (B, S, 2)
        span_labels = batch["span_labels"]    # (B, S)
        span_mask = batch["span_mask"]        # (B, S)

        for b in range(B):
            pred_set = _greedy_decode_non_overlapping(
                logits[b], span_indices[b], span_mask[b],
                margin_threshold=margin_threshold,
            )

            gold_set = set()
            for s_idx in range(span_indices.size(1)):
                if span_mask[b, s_idx].item() < 0.5:
                    continue
                gold_type_id = span_labels[b, s_idx].item()
                if gold_type_id != 0:
                    gold_type = ID_TO_ENTITY_TYPE.get(gold_type_id, "O")
                    if gold_type != "O":
                        start = span_indices[b, s_idx, 0].item()
                        end = span_indices[b, s_idx, 1].item()
                        gold_set.add((start, end, gold_type))

            total_pred_entities += len(pred_set)
            total_gold_entities += len(gold_set)

            all_pred_spans.append(pred_set)
            all_gold_spans.append(gold_set)

    logger.info(
        f"Eval stats: {len(all_pred_spans)} samples, "
        f"total_pred_entities={total_pred_entities}, "
        f"total_gold_entities={total_gold_entities}"
    )
    if total_pred_entities == 0 and total_gold_entities > 0:
        logger.warning(
            "Model predicted 0 entities across all samples! "
            "F1 will be 0. Check if training loss is converging properly."
        )

    return compute_span_f1(all_pred_spans, all_gold_spans)


@torch.no_grad()
def tune_threshold_on_dev(
    model: ImaginEModel,
    dataloader: DataLoader,
    config: ImaginEConfig,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> tuple[float, dict]:
    """Search for the best margin_threshold on the dev set.

    Caches all logits in a single forward pass, then sweeps thresholds
    without re-running inference.

    Returns:
        (best_threshold, best_metrics)
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(201)]  # 0.00 .. 2.00

    model.eval()

    cached_samples: list[dict] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast("cuda", enabled=config.train.fp16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                span_indices=batch["span_indices"],
            )
        logits = outputs["logits"].cpu()
        span_indices = batch["span_indices"].cpu()
        span_labels = batch["span_labels"].cpu()
        span_mask = batch["span_mask"].cpu()

        preds = logits.argmax(dim=-1)
        scores = logits.max(dim=-1).values
        margins = logits[:, :, 1:].max(dim=-1).values - logits[:, :, 0]

        for b in range(logits.size(0)):
            candidates = []
            gold_set = set()
            for s_idx in range(span_indices.size(1)):
                if span_mask[b, s_idx].item() < 0.5:
                    continue

                gold_type_id = span_labels[b, s_idx].item()
                if gold_type_id != 0:
                    gold_type = ID_TO_ENTITY_TYPE.get(gold_type_id, "O")
                    if gold_type != "O":
                        start = span_indices[b, s_idx, 0].item()
                        end = span_indices[b, s_idx, 1].item()
                        gold_set.add((start, end, gold_type))

                pred_id = preds[b, s_idx].item()
                if pred_id == 0:
                    continue
                pred_type = ID_TO_ENTITY_TYPE.get(pred_id, "O")
                if pred_type == "O":
                    continue
                start = span_indices[b, s_idx, 0].item()
                end = span_indices[b, s_idx, 1].item()
                candidates.append((
                    scores[b, s_idx].item(),
                    margins[b, s_idx].item(),
                    start,
                    end,
                    pred_type,
                ))

            candidates.sort(key=lambda x: x[0], reverse=True)
            cached_samples.append({
                "candidates": candidates,
                "gold": gold_set,
            })

    def _eval_with_threshold(thr: float) -> dict:
        all_pred, all_gold = [], []
        for sample in cached_samples:
            pred_set = set()
            occupied = set()
            for _score, margin, start, end, pred_type in sample["candidates"]:
                if margin < thr:
                    continue
                span_positions = range(start, end + 1)
                if any(pos in occupied for pos in span_positions):
                    continue
                occupied.update(span_positions)
                pred_set.add((start, end, pred_type))
            all_pred.append(pred_set)
            all_gold.append(sample["gold"])
        return compute_span_f1(all_pred, all_gold)

    best_thr, best_f1, best_metrics = 0.0, -1.0, {}
    for thr in thresholds:
        m = _eval_with_threshold(thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            best_metrics = m
    logger.info(
        f"Threshold search: best={best_thr:.2f} -> "
        f"P={best_metrics['precision']:.4f} R={best_metrics['recall']:.4f} "
        f"F1={best_metrics['f1']:.4f}"
    )
    return best_thr, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ImaginE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="twitter2017")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--image_dir", type=str, default="./data/images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_model", type=str, default=None,
                        help="Text encoder model name")
    parser.add_argument("--image_model", type=str, default=None,
                        help="Image encoder model name")
    parser.add_argument("--shared_dim", type=int, default=None,
                        help="Shared projection dimension")
    parser.add_argument("--max_span_length", type=int, default=None,
                        help="Maximum candidate span length in words")
    args = parser.parse_args()

    config = ImaginEConfig()
    config.train.device = args.device
    config.train.fp16 = False

    set_seed(42)
    setup_logging()
    device = torch.device(args.device)

    # Load checkpoint config first so processor uses the correct model names
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_config" in ckpt:
        from dataclasses import fields
        saved_cfg = ckpt["model_config"]
        for f in fields(config.model):
            if f.name in saved_cfg:
                setattr(config.model, f.name, saved_cfg[f.name])
        logger.info(f"Loaded model config from checkpoint: {saved_cfg}")
    # CLI overrides take precedence over saved config
    if args.text_model is not None:
        config.model.text_model_name = args.text_model
    if args.image_model is not None:
        config.model.image_model_name = args.image_model
    if args.shared_dim is not None:
        config.model.shared_dim = args.shared_dim
    if args.max_span_length is not None:
        config.model.max_span_length = args.max_span_length

    processor = MNERProcessor(
        text_model_name=config.model.text_model_name,
        image_model_name=config.model.image_model_name,
        max_seq_length=config.model.max_seq_length,
        max_span_length=config.model.max_span_length,
    )

    dataset = TwitterMNERDataset(
        data_file=os.path.join(args.data_dir, args.dataset, f"{args.split}.txt"),
        image_dir=args.image_dir,
        processor=processor,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    model = ImaginEModel(config.model).to(device)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        logger.info(f"Loaded EMA weights from {args.checkpoint}")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    logger.info(f"Evaluating on {args.dataset}/{args.split} ({len(dataset)} samples)")

    metrics = evaluate_model(model, dataloader, config, device)

    logger.info(f"{'='*50}")
    logger.info(
        f"Overall | P: {metrics['precision']:.4f} "
        f"R: {metrics['recall']:.4f} "
        f"F1: {metrics['f1']:.4f}"
    )
    logger.info(f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']}")
    for etype, ef1 in metrics["per_type_f1"].items():
        logger.info(f"  {etype}: F1={ef1:.4f}")


if __name__ == "__main__":
    main()
