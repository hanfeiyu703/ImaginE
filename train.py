"""
ImaginE training script.

Key features:
  - Dual learning rate: encoder 2e-5, new modules 1e-4
  - AdamW with weight decay
  - Linear warmup + cosine decay schedule
  - Best checkpoint selection by dev F1
  - Mixed precision (FP16) support
"""

import os
import sys
import math
import json
import time
import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from config import (
    ImaginEConfig,
    TrainConfig,
    get_coarse_entity_types,
    get_default_dataset,
    get_entity_types,
    get_fine_to_coarse_ids,
    resolve_dataset_split_file,
)
from models.imagine_model import ImaginEModel
from losses.imagine_loss import ImaginELoss
from data.dataset import TwitterMNERDataset, collate_fn
from data.processor import MNERProcessor
from evaluate import evaluate_model, format_metrics_for_logging, tune_threshold_on_dev
from utils import set_seed, setup_logging, count_parameters

logger = logging.getLogger(__name__)


def symmetric_kl_divergence(
    logits1: torch.Tensor, logits2: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL divergence between two logit distributions on valid spans."""
    valid = mask.bool()
    if not valid.any():
        return torch.tensor(0.0, device=logits1.device, requires_grad=True)
    p = torch.nn.functional.log_softmax(logits1[valid], dim=-1)
    q = torch.nn.functional.log_softmax(logits2[valid], dim=-1)
    kl_pq = torch.nn.functional.kl_div(q, p, log_target=True, reduction="batchmean")
    kl_qp = torch.nn.functional.kl_div(p, q, log_target=True, reduction="batchmean")
    return (kl_pq + kl_qp) / 2


def auxiliary_loss_scale(epoch: int, warmup_epochs: int) -> float:
    """Linearly ramp register-style auxiliary losses after epoch 1."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch - 1) / float(warmup_epochs)))


def count_label_occurrences(dataset: TwitterMNERDataset, num_types: int) -> list[int]:
    """Count fine labels in the train split for class-balanced losses."""
    counts = [0] * num_types
    label_to_id = dataset.label_to_id
    for sample in dataset.samples:
        for label in sample["labels"]:
            label_id = label_to_id.get(label, 0)
            if 0 <= label_id < num_types:
                counts[label_id] += 1
    counts[0] = max(counts[0], len(dataset.samples))
    return [max(count, 1) for count in counts]


class ModelEMA:
    """Exponential Moving Average of model parameters for better generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA shadow params (save originals for restore)."""
        self.backup = {}
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.backup[name] = param.clone()
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        """Restore original model params after EMA evaluation."""
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}


def setup_distributed():
    """Initialize DDP if launched via torchrun; fall back to single-GPU otherwise."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model):
    """Return the raw model whether or not it is wrapped in DDP."""
    return model.module if isinstance(model, DDP) else model


def warmup_imagination(
    model: ImaginEModel,
    dataloader: DataLoader,
    config: TrainConfig,
    num_epochs: int = 5,
    num_rev_epochs: int = 0,
) -> None:
    """Pre-train ImaginationPredictors with MSE objectives.

    Forward and reverse warmup epochs are independently controlled.
    Reverse warmup defaults to 0 because SpanVisualAttention leaks z_span
    information into z_v_span, causing the reverse predictor to learn a
    shortcut that collapses once encoders start updating in full training.

    Uses a two-phase approach for speed:
      Phase 1: Pre-compute all encoder outputs once (no_grad, single pass)
      Phase 2: Train imagination modules on cached features (no encoder cost)
    """
    device = config.device
    use_amp = config.fp16
    train_rev = num_rev_epochs > 0

    # --- Phase 1: cache encoder outputs ---
    logger.info("=== Imagination Warmup: caching encoder features ===")
    cached_features = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_indices = batch["span_indices"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            region_features = batch["region_features"].to(device)
            region_mask = batch["region_mask"].to(device)
            with autocast("cuda", enabled=use_amp):
                z_span, _ = model.text_encoder(
                    input_ids, attention_mask,
                    span_indices,
                )
                z_v, z_v_cls = model.image_encoder(
                    pixel_values=pixel_values,
                    region_features=region_features,
                    region_mask=region_mask,
                )
                visual_region_mask = region_mask
                if visual_region_mask is not None and visual_region_mask.size(1) != z_v.size(1):
                    visual_region_mask = None
                if getattr(model, "use_dream_registers", False):
                    visual_registers = model.visual_register_generator(
                        z_v,
                        visual_region_mask,
                    )
                    z_v = model.register_augmented_visual(
                        z_v,
                        visual_registers,
                        visual_region_mask,
                    )
                    if visual_region_mask is not None:
                        weights = visual_region_mask.to(z_v.dtype).unsqueeze(-1)
                        denom = weights.sum(dim=1).clamp(min=1.0)
                        z_v_cls = (z_v * weights).sum(dim=1) / denom
                    else:
                        z_v_cls = z_v.mean(dim=1)
                entry = {
                    "z_span": z_span.float().cpu(),
                    "z_v_cls": z_v_cls.float().cpu(),
                    "span_labels": batch["span_labels"].cpu(),
                    "span_mask": batch["span_mask"].cpu(),
                }
                if train_rev:
                    z_v_span = model.span_visual_attention(
                        z_span,
                        z_v,
                        region_mask=visual_region_mask,
                    )
                    entry["z_v_span"] = z_v_span.float().cpu()
            cached_features.append(entry)
    logger.info(f"Cached {len(cached_features)} batches of encoder features")

    # --- Phase 2: train imagination modules on cached features ---
    seen: set[int] = set()
    warmup_params: list[nn.Parameter] = []
    param_sources = [model.imagination.parameters()]
    if train_rev:
        param_sources.append(model.reverse_imagination.parameters())
    for p in (p for src in param_sources for p in src):
        if id(p) not in seen:
            warmup_params.append(p)
            seen.add(id(p))
    optimizer = torch.optim.AdamW(warmup_params, lr=5e-4, weight_decay=0.01)
    scaler = GradScaler("cuda") if use_amp else None

    rev_status = f" + Rev {num_rev_epochs}ep" if train_rev else " (Rev skipped)"
    logger.info(f"=== Imagination Warmup: Fwd {num_epochs}ep{rev_status} ===")
    for epoch in range(1, num_epochs + 1):
        do_rev = train_rev and epoch <= num_rev_epochs
        model.imagination.train()
        if do_rev:
            model.reverse_imagination.train()
        total_fwd_loss = 0.0
        total_rev_loss = 0.0
        num_steps = 0

        for cached in cached_features:
            z_span = cached["z_span"].to(device)
            z_v_cls = cached["z_v_cls"].to(device)
            span_labels = cached["span_labels"].to(device)
            span_mask = cached["span_mask"].to(device)

            with autocast("cuda", enabled=use_amp):
                B, S, d = z_span.shape
                z_span_flat = z_span.reshape(B * S, -1)

                z_imag_flat = model.imagination(z_span_flat)      # (B*S, K, d)
                K = z_imag_flat.size(1)

                span_labels_flat = span_labels.reshape(B * S)
                span_mask_flat = span_mask.reshape(B * S)
                entity_mask = (span_labels_flat != 0) & (span_mask_flat > 0.5)

                if not entity_mask.any():
                    continue

                gt_idx = span_labels_flat[entity_mask].unsqueeze(-1).unsqueeze(-1)
                gt_idx = gt_idx.expand(-1, 1, d)  # (N_entity, 1, d)

                z_imag_gt = torch.gather(
                    z_imag_flat[entity_mask], 1, gt_idx
                ).squeeze(1)  # (N_entity, d)
                z_v_target = z_v_cls.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
                z_v_target = z_v_target[entity_mask]
                fwd_loss = nn.functional.mse_loss(z_imag_gt, z_v_target.detach())

                loss = fwd_loss

                rev_loss_val = 0.0
                if do_rev:
                    z_v_span = cached["z_v_span"].to(device)
                    z_v_span_flat = z_v_span.reshape(B * S, -1)
                    z_imag_text_flat = model.reverse_imagination(z_v_span_flat)
                    z_imag_text_gt = torch.gather(
                        z_imag_text_flat[entity_mask], 1, gt_idx
                    ).squeeze(1)
                    z_span_target = z_span_flat[entity_mask]
                    rev_loss = nn.functional.mse_loss(z_imag_text_gt, z_span_target.detach())
                    loss = loss + rev_loss
                    rev_loss_val = rev_loss.item()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            total_fwd_loss += fwd_loss.item()
            total_rev_loss += rev_loss_val
            num_steps += 1

        avg_fwd = total_fwd_loss / max(num_steps, 1)
        msg = f"Imagination Warmup Epoch {epoch}/{num_epochs} | Fwd MSE: {avg_fwd:.6f}"
        if do_rev:
            avg_rev = total_rev_loss / max(num_steps, 1)
            msg += f" | Rev MSE: {avg_rev:.6f}"
        logger.info(msg)

    del cached_features
    logger.info("=== Imagination Warmup Complete ===")


def build_optimizer(model: ImaginEModel, config: TrainConfig):
    """Build optimizer with dual learning rates."""
    encoder_params = model.get_encoder_params()
    new_params = model.get_new_module_params()

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": new_params, "lr": config.new_module_lr},
        ],
        weight_decay=config.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, num_training_steps: int, config: TrainConfig):
    """Build learning rate scheduler with warmup."""
    warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def train_epoch(
    model: ImaginEModel,
    criterion: ImaginELoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
    config: TrainConfig,
    epoch: int,
    ema: ModelEMA | None = None,
    r_drop_alpha: float = 0.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    device = config.device
    raw_model = _unwrap(model)

    total_loss = 0.0
    loss_components = {
        "l_task": 0.0, "l_ground": 0.0, "l_ira": 0.0, "l_ico": 0.0, "l_sig": 0.0,
        "l_ira_rev": 0.0, "l_ico_rev": 0.0, "l_sig_rev": 0.0,
        "l_groundable": 0.0, "l_register": 0.0, "l_qsp": 0.0,
        "l_coarse": 0.0, "l_coarse_fine": 0.0,
        "l_no_region_consistency": 0.0, "l_clip_patch": 0.0,
        "l_region_hard_neg": 0.0, "l_set_aux": 0.0,
    }
    total_kl = 0.0
    num_steps = 0
    step = -1
    use_amp = config.fp16 and scaler is not None

    for step, batch in enumerate(dataloader):
        fwd_kwargs = dict(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            region_features=batch["region_features"].to(device),
            region_mask=batch["region_mask"].to(device),
            span_indices=batch["span_indices"].to(device),
            span_labels=batch["span_labels"].to(device),
            knowledge_input_ids=batch["knowledge_input_ids"].to(device),
            knowledge_attention_mask=batch["knowledge_attention_mask"].to(device),
        )
        loss_kwargs = dict(
            labels=batch["span_labels"].to(device),
            span_mask=batch["span_mask"].to(device),
            grounding_labels=batch["grounding_labels"].to(device),
            groundable_labels=batch["groundable_labels"].to(device),
            coarse_labels=batch["coarse_labels"].to(device),
            clip_patch_labels=batch["clip_patch_labels"].to(device),
            task=config.task,
        )

        with autocast("cuda", enabled=use_amp):
            outputs = model(**fwd_kwargs)
            loss_dict = criterion(
                logits=outputs["logits"],
                z_imag=outputs["z_imag"],
                z_attended=outputs["z_attended"],
                scores=outputs["scores"],
                z_imag_text=outputs["z_imag_text"],
                reverse_scores=outputs["reverse_scores"],
                z_span=outputs["z_span"],
                grounding_logits=outputs["grounding_logits"],
                groundable_logits=outputs.get("groundable_logits"),
                coarse_logits=outputs.get("coarse_logits"),
                register_summary=outputs.get("register_summary"),
                clip_patch_logits=outputs.get("clip_patch_logits"),
                set_aux_outputs=outputs.get("set_aux_outputs"),
                visual_gate=outputs.get("visual_relevance"),
                span_indices=batch["span_indices"].to(device),
                **loss_kwargs,
            )

            if r_drop_alpha > 0:
                outputs2 = model(**fwd_kwargs)
                loss_dict2 = criterion(
                    logits=outputs2["logits"],
                    z_imag=outputs2["z_imag"],
                    z_attended=outputs2["z_attended"],
                    scores=outputs2["scores"],
                    z_imag_text=outputs2["z_imag_text"],
                    reverse_scores=outputs2["reverse_scores"],
                    z_span=outputs2["z_span"],
                    grounding_logits=outputs2["grounding_logits"],
                    groundable_logits=outputs2.get("groundable_logits"),
                    coarse_logits=outputs2.get("coarse_logits"),
                    register_summary=outputs2.get("register_summary"),
                    clip_patch_logits=outputs2.get("clip_patch_logits"),
                    set_aux_outputs=outputs2.get("set_aux_outputs"),
                    visual_gate=outputs2.get("visual_relevance"),
                    span_indices=batch["span_indices"].to(device),
                    **loss_kwargs,
                )
                kl = symmetric_kl_divergence(
                    outputs["logits"], outputs2["logits"], batch["span_mask"].to(device),
                )
                combined = (loss_dict["loss"] + loss_dict2["loss"]) / 2 + r_drop_alpha * kl
                total_kl += kl.item()
            else:
                combined = loss_dict["loss"]

        loss = combined / config.gradient_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(raw_model)

        all_comp_names = [
            "loss", "l_task", "l_ground", "l_ira", "l_ico", "l_sig",
            "l_ira_rev", "l_ico_rev", "l_sig_rev",
            "l_groundable", "l_register", "l_qsp",
            "l_coarse", "l_coarse_fine",
            "l_no_region_consistency", "l_clip_patch",
            "l_region_hard_neg", "l_set_aux",
        ]
        dicts_to_check = [("fwd", loss_dict)]
        if r_drop_alpha > 0:
            dicts_to_check.append(("rdrop", loss_dict2))
        for tag, ld in dicts_to_check:
            for comp_name in all_comp_names:
                val = ld[comp_name].item()
                if math.isnan(val) or math.isinf(val):
                    vals_str = ", ".join(
                        f"{k}={ld[k].item()}" for k in all_comp_names
                    )
                    raise RuntimeError(
                        f"NaN/Inf detected in {tag}/{comp_name} at epoch "
                        f"{epoch} step {step+1}! Values: {vals_str}"
                    )
        if r_drop_alpha > 0:
            kl_val = kl.item()
            if math.isnan(kl_val) or math.isinf(kl_val):
                raise RuntimeError(
                    f"NaN/Inf in KL divergence at epoch {epoch} step {step+1}! "
                    f"kl={kl_val}"
                )

        l_task_val = loss_dict["l_task"].item()
        l_sig_val = loss_dict["l_sig"].item()
        if l_task_val > 0 and l_sig_val > 10 * l_task_val:
            logger.warning(
                f"SIGReg magnitude warning at epoch {epoch} step {step+1}: "
                f"l_sig ({l_sig_val:.4f}) > 10 * l_task ({l_task_val:.4f})"
            )

        total_loss += combined.item()
        for k in loss_components:
            loss_components[k] += loss_dict[k].item()
        num_steps += 1

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / num_steps
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_new = optimizer.param_groups[1]["lr"]
            kl_str = f" | L_kl: {total_kl/num_steps:.4f}" if r_drop_alpha > 0 else ""
            logger.info(
                f"Epoch {epoch} Step {step+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"L_task: {loss_components['l_task']/num_steps:.4f} | "
                f"L_ground: {loss_components['l_ground']/num_steps:.4f} | "
                f"L_gable: {loss_components['l_groundable']/num_steps:.4f} | "
                f"L_ira: {loss_components['l_ira']/num_steps:.4f} | "
                f"L_ico: {loss_components['l_ico']/num_steps:.4f} | "
                f"L_sig: {loss_components['l_sig']/num_steps:.4f} | "
                f"L_ira_r: {loss_components['l_ira_rev']/num_steps:.4f} | "
                f"L_ico_r: {loss_components['l_ico_rev']/num_steps:.4f} | "
                f"L_sig_r: {loss_components['l_sig_rev']/num_steps:.4f} | "
                f"L_reg: {loss_components['l_register']/num_steps:.4f} | "
                f"L_qsp: {loss_components['l_qsp']/num_steps:.4f} | "
                f"L_coarse: {loss_components['l_coarse']/num_steps:.4f} | "
                f"L_cf: {loss_components['l_coarse_fine']/num_steps:.4f} | "
                f"L_noreg: {loss_components['l_no_region_consistency']/num_steps:.4f} | "
                f"L_patch: {loss_components['l_clip_patch']/num_steps:.4f} | "
                f"L_hneg: {loss_components['l_region_hard_neg']/num_steps:.4f} | "
                f"L_set: {loss_components['l_set_aux']/num_steps:.4f}{kl_str} | "
                f"LR_enc: {lr_enc:.2e} LR_new: {lr_new:.2e}"
            )

    if num_steps > 0 and (step + 1) % config.gradient_accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(raw_model)

    return {
        "loss": total_loss / max(num_steps, 1),
        **{k: v / max(num_steps, 1) for k, v in loss_components.items()},
    }


def main():
    # --- Distributed setup (no-op when launched with plain python) ---
    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    parser = argparse.ArgumentParser(description="Train ImaginE")
    parser.add_argument("--task", type=str, default="mner",
                        choices=["mner", "gmner", "fmnerg"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name. Defaults to a task-specific dataset.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--image_dir", type=str, default="./data/images")
    parser.add_argument("--vinvl_dir", type=str, default=None,
                        help="Directory containing VinVL .npz features")
    parser.add_argument("--annotation_dir", type=str, default=None,
                        help="Directory containing XML grounding annotations")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--new_module_lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_model", type=str, default=None,
                        help="Text encoder model name")
    parser.add_argument("--image_model", type=str, default=None,
                        help="Image encoder model name")
    parser.add_argument("--visual_backend", type=str, default="raw_image",
                        choices=["raw_image", "vinvl"])
    parser.add_argument("--shared_dim", type=int, default=None,
                        help="Shared projection dimension (default: 384)")
    parser.add_argument("--max_regions", type=int, default=36,
                        help="Number of VinVL proposals to keep")
    parser.add_argument("--vinvl_feature_dim", type=int, default=2048,
                        help="Dimensionality of VinVL region features")
    parser.add_argument("--normalize_vinvl", action="store_true",
                        help="Apply reference-style global normalization to VinVL features")
    parser.add_argument("--warmup_imagination_epochs", type=int, default=0,
                        help="Epochs to warmup ImaginationPredictor before full training")
    parser.add_argument("--warmup_rev_epochs", type=int, default=0,
                        help="Epochs to warmup ReverseImaginationPredictor (0=skip)")
    parser.add_argument("--init_type_embed_from_clip", action="store_true",
                        help="Initialize type embeddings from CLIP text encoder")
    parser.add_argument("--r_drop_alpha", type=float, default=0.5,
                        help="R-Drop KL divergence weight (0 = disabled)")
    parser.add_argument("--alpha_rev", type=float, default=1.0,
                        help="Reverse L_ira weight")
    parser.add_argument("--beta_rev", type=float, default=0.5,
                        help="Reverse L_ico weight")
    parser.add_argument("--gamma_rev", type=float, default=0.05,
                        help="Reverse L_sig weight")
    parser.add_argument("--grounding_weight", type=float, default=1.0,
                        help="Weight for the grounding supervision term")
    parser.add_argument("--groundable_weight", type=float, default=0.0,
                        help="Weight for the binary groundable auxiliary term")
    parser.add_argument("--use_groundable_gate", action="store_true",
                        help="Use groundable logits as a hard gate during evaluation")
    parser.add_argument("--use_region_pointer", action="store_true",
                        help="Use H-Index-style dot-product region pointer grounding")
    parser.add_argument("--grounding_decode_mode", type=str, default="argmax",
                        choices=["argmax", "soft_groundable", "hard_groundable"],
                        help="How to decode grounding predictions at evaluation time")
    parser.add_argument("--groundable_threshold", type=float, default=0.0,
                        help="Groundable logit threshold used by soft/hard grounding decode")
    parser.add_argument("--tune_grounding_decode", action="store_true",
                        help="Tune entity margin and groundable threshold jointly on dev")
    parser.add_argument("--use_dream_registers", action="store_true",
                        help="Enable lightweight DreamPRVR-style visual registers (VinVL only)")
    parser.add_argument("--num_visual_registers", type=int, default=4,
                        help="Number of global visual registers")
    parser.add_argument("--use_span_type_registers", action="store_true",
                        help="Let each span/type imagination query read region+global registers")
    parser.add_argument("--register_weight", type=float, default=0.1,
                        help="Weight for register-text InfoNCE auxiliary loss")
    parser.add_argument("--qsp_weight", type=float, default=0.05,
                        help="Weight for lightweight QSP auxiliary loss")
    parser.add_argument("--coarse_weight", type=float, default=0.0,
                        help="Weight for FMNERG coarse type auxiliary loss")
    parser.add_argument("--coarse_fine_weight", type=float, default=0.0,
                        help="Weight for FMNERG coarse-fine consistency loss")
    parser.add_argument("--use_hierarchical_fine_logits", action="store_true",
                        help="Inject coarse-type log priors into FMNERG fine logits")
    parser.add_argument("--coarse_prior_weight", type=float, default=0.3,
                        help="Weight for TIGER-style coarse prior fine-logit calibration")
    parser.add_argument("--use_coarse_fine_transition", action="store_true",
                        help="Use a taxonomy transition matrix for FMNERG fine-logit calibration")
    parser.add_argument("--transition_prior_weight", type=float, default=0.0,
                        help="Weight for taxonomy transition fine-logit calibration")
    parser.add_argument("--no_region_consistency_weight", type=float, default=0.0,
                        help="Weight for no-region / groundable consistency loss")
    parser.add_argument("--caption_files", nargs="*", default=[],
                        help="Optional image caption files to append as text context")
    parser.add_argument("--append_caption", action="store_true",
                        help="Append loaded image captions to text encoder input")
    parser.add_argument("--caption_max_words", type=int, default=32,
                        help="Maximum caption words appended per sample")
    parser.add_argument("--use_clip_patch_fallback", action="store_true",
                        help="Train/evaluate frozen-CLIP patch fallback grounding")
    parser.add_argument("--clip_patch_weight", type=float, default=0.0,
                        help="Weight for CLIP patch grounding auxiliary loss")
    parser.add_argument("--clip_fallback_threshold", type=float, default=0.0,
                        help="Groundable logit threshold for CLIP patch fallback decode")
    parser.add_argument("--knowledge_injection", type=str, default="off",
                        choices=["off", "gated_span"],
                        help="Inject independently encoded knowledge into span typing heads")
    parser.add_argument("--knowledge_files", nargs="*", default=[],
                        help="Optional image knowledge/caption files for gated span injection")
    parser.add_argument("--knowledge_max_words", type=int, default=32,
                        help="Maximum knowledge words encoded per sample")
    parser.add_argument("--knowledge_dropout", type=float, default=0.2,
                        help="Dropout applied to gated knowledge residuals")
    parser.add_argument("--knowledge_gate_init", type=float, default=-2.0,
                        help="Initial knowledge gate bias")
    parser.add_argument("--fine_loss_type", type=str, default="ce",
                        choices=["ce", "cb_focal"],
                        help="Fine-type loss for FMNERG")
    parser.add_argument("--fine_focal_gamma", type=float, default=1.5,
                        help="Focal gamma for --fine_loss_type cb_focal")
    parser.add_argument("--fine_class_balance_beta", type=float, default=0.999,
                        help="Class-balanced effective-number beta")
    parser.add_argument("--use_fine_rerank", action="store_true",
                        help="Tune a dev-time parent coarse-score rerank for FMNERG")
    parser.add_argument("--fine_rerank_lambdas", nargs="*", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.5],
                        help="Lambda grid for FMNERG fine reranking")
    parser.add_argument("--use_type_aware_region_pointer", action="store_true",
                        help="Condition the region pointer query on entity type embeddings")
    parser.add_argument("--region_hard_negative_weight", type=float, default=0.0,
                        help="Weight for grounding hard-negative region margin loss")
    parser.add_argument("--use_set_prediction_aux", action="store_true",
                        help="Enable MQSPN-style fixed-query recall auxiliary head")
    parser.add_argument("--set_aux_weight", type=float, default=0.0,
                        help="Weight for fixed-query set prediction auxiliary loss")
    parser.add_argument("--set_aux_queries", type=int, default=60,
                        help="Number of fixed queries for set prediction auxiliary loss")
    parser.add_argument("--set_aux_warmup_epochs", type=int, default=1,
                        help="Epochs to ramp set auxiliary loss")
    parser.add_argument("--aux_warmup_epochs", type=int, default=0,
                        help="Epochs to ramp register/qsp/coarse-fine auxiliary weights")
    parser.add_argument("--ema_decay", type=float, default=0.995,
                        help="EMA decay rate (lower = faster tracking)")
    args = parser.parse_args()

    # --- Build config ---
    config = ImaginEConfig()
    config.train.task = args.task
    config.train.dataset = args.dataset or get_default_dataset(args.task)
    config.train.data_dir = args.data_dir
    config.train.image_dir = args.image_dir
    config.train.vinvl_dir = args.vinvl_dir
    config.train.annotation_dir = args.annotation_dir
    config.train.output_dir = args.output_dir
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.num_workers = args.num_workers
    config.train.encoder_lr = args.encoder_lr
    config.train.new_module_lr = args.new_module_lr
    config.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.train.seed = args.seed
    config.train.fp16 = args.fp16
    config.train.device = args.device
    config.train.normalize_vinvl = args.normalize_vinvl
    config.train.aux_warmup_epochs = args.aux_warmup_epochs
    config.train.set_aux_warmup_epochs = args.set_aux_warmup_epochs
    config.train.tune_grounding_decode = args.tune_grounding_decode
    config.train.append_caption = args.append_caption
    config.train.caption_files = args.caption_files
    config.train.caption_max_words = args.caption_max_words
    config.train.knowledge_files = args.knowledge_files
    config.train.knowledge_max_words = args.knowledge_max_words
    config.train.knowledge_dropout = args.knowledge_dropout
    config.train.use_fine_rerank = args.use_fine_rerank
    config.train.fine_rerank_lambdas = args.fine_rerank_lambdas
    config.model.task = args.task
    config.model.visual_backend = args.visual_backend
    config.model.max_regions = args.max_regions
    config.model.vinvl_feature_dim = args.vinvl_feature_dim
    config.model.use_dream_registers = args.use_dream_registers
    config.model.num_visual_registers = args.num_visual_registers
    config.model.use_span_type_registers = args.use_span_type_registers
    config.model.use_groundable_gate = args.use_groundable_gate
    config.model.use_region_pointer = args.use_region_pointer
    config.model.grounding_decode_mode = args.grounding_decode_mode
    config.model.groundable_threshold = args.groundable_threshold
    config.model.use_hierarchical_fine_logits = args.use_hierarchical_fine_logits
    config.model.coarse_prior_weight = args.coarse_prior_weight
    config.model.use_coarse_fine_transition = args.use_coarse_fine_transition
    config.model.transition_prior_weight = args.transition_prior_weight
    config.model.use_clip_patch_fallback = args.use_clip_patch_fallback
    config.model.clip_fallback_threshold = args.clip_fallback_threshold
    config.model.knowledge_injection = args.knowledge_injection
    config.model.knowledge_dropout = args.knowledge_dropout
    config.model.knowledge_gate_init = args.knowledge_gate_init
    config.model.use_type_aware_region_pointer = args.use_type_aware_region_pointer
    config.model.use_set_prediction_aux = args.use_set_prediction_aux
    config.model.set_aux_queries = args.set_aux_queries
    config.model.num_types = len(get_entity_types(args.task))
    config.model.num_coarse_types = len(get_coarse_entity_types(args.task))
    config.loss.alpha = args.alpha
    config.loss.beta = args.beta
    config.loss.gamma = args.gamma
    config.loss.tau = args.tau
    config.loss.r_drop_alpha = args.r_drop_alpha
    config.loss.alpha_rev = args.alpha_rev
    config.loss.beta_rev = args.beta_rev
    config.loss.gamma_rev = args.gamma_rev
    config.loss.grounding_weight = args.grounding_weight
    config.loss.groundable_weight = args.groundable_weight
    config.loss.register_weight = args.register_weight
    config.loss.qsp_weight = args.qsp_weight
    config.loss.coarse_weight = args.coarse_weight
    config.loss.coarse_fine_weight = args.coarse_fine_weight
    config.loss.no_region_consistency_weight = args.no_region_consistency_weight
    config.loss.clip_patch_weight = args.clip_patch_weight
    config.loss.region_hard_negative_weight = args.region_hard_negative_weight
    config.loss.fine_loss_type = args.fine_loss_type
    config.loss.fine_focal_gamma = args.fine_focal_gamma
    config.loss.fine_class_balance_beta = args.fine_class_balance_beta
    config.loss.set_aux_weight = args.set_aux_weight
    if args.text_model is not None:
        config.model.text_model_name = args.text_model
    if args.image_model is not None:
        config.model.image_model_name = args.image_model
    if args.shared_dim is not None:
        config.model.shared_dim = args.shared_dim
    if args.use_dream_registers and args.visual_backend != "vinvl":
        raise ValueError("Dream registers currently require --visual_backend vinvl.")
    if args.use_dream_registers and args.num_visual_registers < 1:
        raise ValueError("--num_visual_registers must be at least 1.")
    if args.use_span_type_registers and not args.use_dream_registers:
        raise ValueError("--use_span_type_registers requires --use_dream_registers.")
    if args.use_region_pointer and args.visual_backend != "vinvl":
        raise ValueError("--use_region_pointer requires --visual_backend vinvl.")
    if args.use_hierarchical_fine_logits and args.task != "fmnerg":
        raise ValueError("--use_hierarchical_fine_logits is only valid for --task fmnerg.")
    if args.use_coarse_fine_transition and args.task != "fmnerg":
        raise ValueError("--use_coarse_fine_transition is only valid for --task fmnerg.")
    if args.use_clip_patch_fallback and args.visual_backend != "vinvl":
        raise ValueError("--use_clip_patch_fallback requires --visual_backend vinvl.")
    if args.knowledge_injection == "gated_span" and not args.knowledge_files:
        raise ValueError("--knowledge_injection gated_span requires --knowledge_files.")
    if args.fine_loss_type != "ce" and args.task != "fmnerg":
        raise ValueError("--fine_loss_type cb_focal is only valid for --task fmnerg.")
    if args.use_fine_rerank and args.task != "fmnerg":
        raise ValueError("--use_fine_rerank is only valid for --task fmnerg.")
    if args.use_type_aware_region_pointer and not args.use_region_pointer:
        raise ValueError("--use_type_aware_region_pointer requires --use_region_pointer.")
    if args.use_set_prediction_aux and args.set_aux_queries < 1:
        raise ValueError("--set_aux_queries must be at least 1.")
    if args.task in {"gmner", "fmnerg"} and args.visual_backend != "vinvl":
        raise ValueError("GMNER/FMNERG currently require --visual_backend vinvl.")
    if args.visual_backend == "vinvl" and (not args.vinvl_dir or not args.annotation_dir):
        raise ValueError("VinVL backend requires both --vinvl_dir and --annotation_dir.")

    # --- Setup ---
    set_seed(config.train.seed + rank)
    os.makedirs(config.train.output_dir, exist_ok=True)
    if is_main:
        setup_logging(os.path.join(config.train.output_dir, "train.log"))
    else:
        logging.basicConfig(level=logging.WARNING)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else args.device)
    else:
        device = torch.device("cpu" if args.device.startswith("cuda") else args.device)
    config.train.device = str(device)
    config.train.fp16 = config.train.fp16 and device.type == "cuda"
    if is_main:
        logger.info(f"Config: {config}")
        logger.info(f"World size: {world_size}")

    # --- Data ---
    processor = MNERProcessor(
        text_model_name=config.model.text_model_name,
        image_model_name=config.model.image_model_name,
        max_seq_length=config.model.max_seq_length,
        max_span_length=config.model.max_span_length,
        visual_backend=config.model.visual_backend,
        load_images_for_vinvl=config.model.use_clip_patch_fallback,
    )

    train_file = resolve_dataset_split_file(
        data_dir=config.train.data_dir,
        dataset=config.train.dataset,
        split="train",
        task=config.train.task,
    )
    dev_file = resolve_dataset_split_file(
        data_dir=config.train.data_dir,
        dataset=config.train.dataset,
        split="dev",
        task=config.train.task,
    )
    test_file = resolve_dataset_split_file(
        data_dir=config.train.data_dir,
        dataset=config.train.dataset,
        split="test",
        task=config.train.task,
    )

    train_dataset = TwitterMNERDataset(
        data_file=train_file,
        image_dir=config.train.image_dir,
        processor=processor,
        is_train=True,
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
    dev_dataset = TwitterMNERDataset(
        data_file=dev_file,
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
    test_dataset = TwitterMNERDataset(
        data_file=test_file,
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.train.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=config.train.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.train.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=config.train.num_workers, pin_memory=True,
    )

    if is_main:
        for split_name, ds in [("train", train_dataset), ("dev", dev_dataset), ("test", test_dataset)]:
            entity_count = sum(
                1 for s in ds.samples for l in s["labels"] if l != "O"
            )
            logger.info(
                f"  {split_name}: {len(ds)} samples, {entity_count} entity tokens"
            )
        logger.info(
            f"Data loaded: train={len(train_dataset)}, "
            f"dev={len(dev_dataset)}, test={len(test_dataset)}"
        )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty! Check data path: "
            f"{train_file}"
        )

    # --- Model (warmup runs BEFORE DDP wrapping for simplicity) ---
    model = ImaginEModel(config.model).to(device)
    if args.init_type_embed_from_clip:
        model.init_type_embeddings_from_clip()
    if is_main:
        enc_params, new_params = count_parameters(model)
        logger.info(f"Encoder params: {enc_params/1e6:.1f}M, New params: {new_params/1e6:.1f}M")

    if args.warmup_imagination_epochs > 0:
        warmup_imagination(
            model, train_loader, config.train,
            num_epochs=args.warmup_imagination_epochs,
            num_rev_epochs=args.warmup_rev_epochs,
        )

    # --- Wrap model in DDP ---
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank],
            find_unused_parameters=True,   # Required: RoBERTa pooler & CLIP post_layernorm are unused
            broadcast_buffers=False,
        )
    raw_model = _unwrap(model)

    # --- Loss ---
    fine_class_counts = (
        count_label_occurrences(train_dataset, config.model.num_types)
        if config.loss.fine_loss_type == "cb_focal"
        else None
    )
    criterion = ImaginELoss(
        alpha=config.loss.alpha,
        beta=config.loss.beta,
        gamma=config.loss.gamma,
        tau=config.loss.tau,
        label_smoothing=config.loss.label_smoothing,
        num_types=config.model.num_types,
        alpha_rev=config.loss.alpha_rev,
        beta_rev=config.loss.beta_rev,
        gamma_rev=config.loss.gamma_rev,
        grounding_weight=config.loss.grounding_weight,
        groundable_weight=config.loss.groundable_weight,
        register_weight=config.loss.register_weight,
        qsp_weight=config.loss.qsp_weight,
        num_coarse_types=config.model.num_coarse_types,
        fine_to_coarse_ids=get_fine_to_coarse_ids(config.train.task),
        coarse_weight=config.loss.coarse_weight,
        coarse_fine_weight=config.loss.coarse_fine_weight,
        no_region_consistency_weight=config.loss.no_region_consistency_weight,
        clip_patch_weight=config.loss.clip_patch_weight,
        region_hard_negative_weight=config.loss.region_hard_negative_weight,
        fine_loss_type=config.loss.fine_loss_type,
        fine_focal_gamma=config.loss.fine_focal_gamma,
        fine_class_balance_beta=config.loss.fine_class_balance_beta,
        fine_class_counts=fine_class_counts,
        set_aux_weight=config.loss.set_aux_weight,
    )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(raw_model, config.train)
    num_training_steps = (
        len(train_loader) * config.train.epochs
        // config.train.gradient_accumulation_steps
    )
    scheduler = build_scheduler(optimizer, num_training_steps, config.train)

    scaler = GradScaler("cuda") if config.train.fp16 else None

    # --- EMA ---
    ema = ModelEMA(raw_model, decay=args.ema_decay)

    # --- Training Loop ---
    best_f1 = -1.0
    best_epoch = 0
    best_dev_metrics = None
    epoch_losses = []

    try:
        for epoch in range(1, config.train.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main:
                logger.info(f"{'='*60}")
                logger.info(f"Epoch {epoch}/{config.train.epochs}")

            aux_scale = auxiliary_loss_scale(epoch, config.train.aux_warmup_epochs)
            criterion.register_weight = config.loss.register_weight * aux_scale
            criterion.qsp_weight = config.loss.qsp_weight * aux_scale
            criterion.coarse_fine_weight = config.loss.coarse_fine_weight * aux_scale
            criterion.set_aux_weight = (
                config.loss.set_aux_weight
                * auxiliary_loss_scale(epoch, config.train.set_aux_warmup_epochs)
            )
            if is_main and config.train.aux_warmup_epochs > 0:
                logger.info(
                    "Auxiliary register/qsp/coarse-fine loss scale: %.3f "
                    "(warmup_epochs=%d)",
                    aux_scale,
                    config.train.aux_warmup_epochs,
                )

            t_start = time.time()
            train_metrics = train_epoch(
                model, criterion, train_loader, optimizer, scheduler,
                scaler, config.train, epoch, ema=ema,
                r_drop_alpha=config.loss.r_drop_alpha,
            )
            t_elapsed = time.time() - t_start

            if is_main:
                logger.info(
                    f"Train | Loss: {train_metrics['loss']:.4f} | "
                    f"L_task: {train_metrics['l_task']:.4f} | "
                    f"L_ground: {train_metrics['l_ground']:.4f} | "
                    f"L_gable: {train_metrics['l_groundable']:.4f} | "
                    f"L_ira: {train_metrics['l_ira']:.4f} | "
                    f"L_ico: {train_metrics['l_ico']:.4f} | "
                    f"L_sig: {train_metrics['l_sig']:.4f} | "
                    f"L_ira_r: {train_metrics['l_ira_rev']:.4f} | "
                    f"L_ico_r: {train_metrics['l_ico_rev']:.4f} | "
                    f"L_sig_r: {train_metrics['l_sig_rev']:.4f} | "
                    f"L_reg: {train_metrics['l_register']:.4f} | "
                    f"L_qsp: {train_metrics['l_qsp']:.4f} | "
                    f"L_coarse: {train_metrics['l_coarse']:.4f} | "
                    f"L_cf: {train_metrics['l_coarse_fine']:.4f} | "
                    f"L_noreg: {train_metrics['l_no_region_consistency']:.4f} | "
                    f"L_patch: {train_metrics['l_clip_patch']:.4f} | "
                    f"L_hneg: {train_metrics['l_region_hard_neg']:.4f} | "
                    f"L_set: {train_metrics['l_set_aux']:.4f} | "
                    f"Time: {t_elapsed:.1f}s"
                )
                epoch_losses.append(train_metrics['loss'])

                # --- Dev Evaluation with EMA weights (rank 0 only) ---
                ema.apply_shadow(raw_model)
                dev_metrics = evaluate_model(raw_model, dev_loader, config, device)
                ema.restore(raw_model)

                logger.info(
                    "Dev   | P: %.4f R: %.4f F1: %.4f",
                    dev_metrics["precision"],
                    dev_metrics["recall"],
                    dev_metrics["f1"],
                )
                for line in format_metrics_for_logging(dev_metrics)[1:]:
                    logger.info(line)

                if dev_metrics["f1"] > best_f1:
                    best_f1 = dev_metrics["f1"]
                    best_epoch = epoch
                    best_dev_metrics = dev_metrics
                    save_path = os.path.join(config.train.output_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "best_f1": best_f1,
                        "entity_types": get_entity_types(config.train.task),
                        "model_config": asdict(config.model),
                        "train_config": asdict(config.train),
                        "loss_config": asdict(config.loss),
                    }, save_path)
                    logger.info(f"New best model saved (F1={best_f1:.4f})")

            if world_size > 1:
                dist.barrier()

        if is_main:
            logger.info(f"{'='*60}")
            logger.info(f"Best dev F1: {best_f1:.4f} at epoch {best_epoch}")

            if len(epoch_losses) >= 2:
                if epoch_losses[-1] >= epoch_losses[0]:
                    logger.warning(
                        f"Loss did NOT decrease over training: "
                        f"epoch 1 loss={epoch_losses[0]:.4f}, "
                        f"final epoch loss={epoch_losses[-1]:.4f}. "
                        f"The model may not be learning — check hyperparameters."
                    )
                else:
                    logger.info(
                        f"Loss trend OK: epoch 1={epoch_losses[0]:.4f} -> "
                        f"final={epoch_losses[-1]:.4f} (decreased)"
                    )

            # --- Final Test Evaluation with best EMA weights (rank 0 only) ---
            best_ckpt = torch.load(
                os.path.join(config.train.output_dir, "best_model.pt"),
                map_location=device,
            )
            if "ema_state_dict" in best_ckpt:
                raw_model.load_state_dict(best_ckpt["ema_state_dict"])
            else:
                raw_model.load_state_dict(best_ckpt["model_state_dict"])

            # Tune margin threshold on dev set, then apply to test
            best_thr, best_groundable_thr, best_fine_lambda, tuned_dev = tune_threshold_on_dev(
                raw_model, dev_loader, config, device,
            )
            logger.info(
                "Using tuned margin_threshold=%.2f groundable_threshold=%.2f fine_lambda=%.2f for test",
                best_thr,
                best_groundable_thr,
                best_fine_lambda,
            )

            test_metrics = evaluate_model(
                raw_model, test_loader, config, device,
                margin_threshold=best_thr,
                groundable_threshold=best_groundable_thr,
                fine_rerank_lambda=best_fine_lambda,
            )
            for line in format_metrics_for_logging(test_metrics):
                logger.info(line.replace(test_metrics["main_metric_name"].upper(), "TEST", 1))

            with open(os.path.join(config.train.output_dir, "results.json"), "w") as f:
                json.dump({
                    "task": config.train.task,
                    "dataset": config.train.dataset,
                    "entity_types": get_entity_types(config.train.task),
                    "best_epoch": best_epoch,
                    "best_margin_threshold": best_thr,
                    "best_groundable_threshold": best_groundable_thr,
                    "best_fine_rerank_lambda": best_fine_lambda,
                    "tuned_dev": tuned_dev,
                    "dev": best_dev_metrics,
                    "test": test_metrics,
                    "epoch_losses": epoch_losses,
                    "dream_registers": {
                        "enabled": config.model.use_dream_registers,
                        "num_visual_registers": config.model.num_visual_registers,
                        "span_type_conditioned": config.model.use_span_type_registers,
                        "register_weight": config.loss.register_weight,
                        "qsp_weight": config.loss.qsp_weight,
                        "aux_warmup_epochs": config.train.aux_warmup_epochs,
                    },
                    "coarse_fine": {
                        "num_coarse_types": config.model.num_coarse_types,
                        "coarse_weight": config.loss.coarse_weight,
                        "coarse_fine_weight": config.loss.coarse_fine_weight,
                        "use_hierarchical_fine_logits": config.model.use_hierarchical_fine_logits,
                        "coarse_prior_weight": config.model.coarse_prior_weight,
                        "use_coarse_fine_transition": config.model.use_coarse_fine_transition,
                        "transition_prior_weight": config.model.transition_prior_weight,
                    },
                    "groundable_head": {
                        "weight": config.loss.groundable_weight,
                        "gate_enabled": config.model.use_groundable_gate,
                    },
                    "region_pointer": {
                        "enabled": config.model.use_region_pointer,
                        "grounding_decode_mode": config.model.grounding_decode_mode,
                        "groundable_threshold": config.model.groundable_threshold,
                        "tune_grounding_decode": config.train.tune_grounding_decode,
                        "tuned_groundable_threshold": best_groundable_thr,
                    },
                    "no_region_consistency": {
                        "weight": config.loss.no_region_consistency_weight,
                    },
                    "caption_context": {
                        "enabled": config.train.append_caption,
                        "caption_files": config.train.caption_files,
                        "caption_max_words": config.train.caption_max_words,
                    },
                    "knowledge_injection": {
                        "mode": config.model.knowledge_injection,
                        "knowledge_files": config.train.knowledge_files,
                        "knowledge_max_words": config.train.knowledge_max_words,
                        "knowledge_dropout": config.train.knowledge_dropout,
                        "knowledge_gate_init": config.model.knowledge_gate_init,
                    },
                    "fine_type_calibration": {
                        "fine_loss_type": config.loss.fine_loss_type,
                        "fine_focal_gamma": config.loss.fine_focal_gamma,
                        "fine_class_balance_beta": config.loss.fine_class_balance_beta,
                        "use_fine_rerank": config.train.use_fine_rerank,
                        "fine_rerank_lambdas": config.train.fine_rerank_lambdas,
                        "best_fine_rerank_lambda": best_fine_lambda,
                    },
                    "grounding_v2": {
                        "use_type_aware_region_pointer": config.model.use_type_aware_region_pointer,
                        "region_hard_negative_weight": config.loss.region_hard_negative_weight,
                    },
                    "set_prediction_aux": {
                        "enabled": config.model.use_set_prediction_aux,
                        "set_aux_queries": config.model.set_aux_queries,
                        "set_aux_weight": config.loss.set_aux_weight,
                        "set_aux_warmup_epochs": config.train.set_aux_warmup_epochs,
                    },
                    "clip_patch_fallback": {
                        "enabled": config.model.use_clip_patch_fallback,
                        "clip_patch_weight": config.loss.clip_patch_weight,
                        "clip_fallback_threshold": config.model.clip_fallback_threshold,
                    },
                }, f, indent=2)

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA Out of Memory! Try reducing --batch_size (current: %d) "
            "or reducing max_spans in the dataset. "
            "With batch_size=4 and max_spans=192, z_v expansion requires ~11.1GB.",
            config.train.batch_size,
        )
        sys.exit(1)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
