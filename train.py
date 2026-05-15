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

from config import ImaginEConfig, ModelConfig, LossConfig, TrainConfig
from models.imagine_model import ImaginEModel
from losses.imagine_loss import ImaginELoss
from data.dataset import TwitterMNERDataset, collate_fn
from data.processor import MNERProcessor
from evaluate import evaluate_model, compute_span_f1, tune_threshold_on_dev
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
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast("cuda", enabled=use_amp):
                z_span, _ = model.text_encoder(
                    batch["input_ids"], batch["attention_mask"],
                    batch["span_indices"],
                )
                z_v, z_v_cls = model.image_encoder(batch["pixel_values"])
                entry = {
                    "z_span": z_span.float().cpu(),
                    "z_v_cls": z_v_cls.float().cpu(),
                    "span_labels": batch["span_labels"].cpu(),
                    "span_mask": batch["span_mask"].cpu(),
                }
                if train_rev:
                    z_v_span = model.span_visual_attention(z_span, z_v)
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
        "l_task": 0.0, "l_ira": 0.0, "l_ico": 0.0, "l_sig": 0.0,
        "l_ira_rev": 0.0, "l_ico_rev": 0.0, "l_sig_rev": 0.0,
    }
    total_kl = 0.0
    num_steps = 0
    step = -1
    use_amp = config.fp16 and scaler is not None

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        fwd_kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            span_indices=batch["span_indices"],
        )
        loss_kwargs = dict(
            labels=batch["span_labels"],
            span_mask=batch["span_mask"],
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
                visual_gate=outputs.get("visual_relevance"),
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
                    visual_gate=outputs2.get("visual_relevance"),
                    **loss_kwargs,
                )
                kl = symmetric_kl_divergence(
                    outputs["logits"], outputs2["logits"], batch["span_mask"],
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
            "loss", "l_task", "l_ira", "l_ico", "l_sig",
            "l_ira_rev", "l_ico_rev", "l_sig_rev",
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
                f"L_ira: {loss_components['l_ira']/num_steps:.4f} | "
                f"L_ico: {loss_components['l_ico']/num_steps:.4f} | "
                f"L_sig: {loss_components['l_sig']/num_steps:.4f} | "
                f"L_ira_r: {loss_components['l_ira_rev']/num_steps:.4f} | "
                f"L_ico_r: {loss_components['l_ico_rev']/num_steps:.4f} | "
                f"L_sig_r: {loss_components['l_sig_rev']/num_steps:.4f}{kl_str} | "
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
    parser.add_argument("--dataset", type=str, default="twitter2017",
                        choices=["twitter2015", "twitter2017"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--image_dir", type=str, default="./data/images")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_schedule_epochs", type=int, default=None,
                        help="Epoch count used only for LR scheduler total steps; defaults to --epochs")
    parser.add_argument("--batch_size", type=int, default=16)
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
    parser.add_argument("--shared_dim", type=int, default=None,
                        help="Shared projection dimension (default: 384)")
    parser.add_argument("--max_span_length", type=int, default=None,
                        help="Maximum candidate span length in words (default: 4)")
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
    parser.add_argument("--ema_decay", type=float, default=0.995,
                        help="EMA decay rate (lower = faster tracking)")
    parser.add_argument("--eval_test_each_epoch", action="store_true",
                        help="Evaluate test set at every epoch and store metrics in results.json")
    parser.add_argument("--tune_threshold_each_epoch", action="store_true",
                        help="Tune margin threshold on dev at every epoch, then evaluate test with that threshold")
    parser.add_argument("--select_best_by", type=str, default="dev_f1",
                        choices=["dev_f1", "test_f1"],
                        help="Metric used to select best checkpoint (test_f1 is diagnostic only)")
    parser.add_argument("--final_eval_checkpoint", type=str, default="best",
                        choices=["best", "last"],
                        help="Checkpoint used for final test evaluation")
    parser.add_argument("--eval_raw_test_final", action="store_true",
                        help="Evaluate the final checkpoint on test with the default threshold before tuned-threshold evaluation")
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping based on dev metrics")
    parser.add_argument("--early_stop_metric", type=str, default="dev_f1",
                        choices=["dev_f1"],
                        help="Metric used for early stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.005,
                        help="Minimum dev F1 improvement required to reset early-stop patience")
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Number of consecutive epochs without meaningful improvement before stopping")
    parser.add_argument("--early_stop_min_epochs", type=int, default=3,
                        help="Minimum number of formal training epochs to run before early stopping can trigger")
    args = parser.parse_args()

    # --- Build config ---
    config = ImaginEConfig()
    config.train.dataset = args.dataset
    config.train.data_dir = args.data_dir
    config.train.image_dir = args.image_dir
    config.train.output_dir = args.output_dir
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.encoder_lr = args.encoder_lr
    config.train.new_module_lr = args.new_module_lr
    config.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.train.seed = args.seed
    config.train.fp16 = args.fp16
    config.train.device = args.device
    config.train.metric_for_best = args.select_best_by
    config.loss.alpha = args.alpha
    config.loss.beta = args.beta
    config.loss.gamma = args.gamma
    config.loss.tau = args.tau
    config.loss.r_drop_alpha = args.r_drop_alpha
    config.loss.alpha_rev = args.alpha_rev
    config.loss.beta_rev = args.beta_rev
    config.loss.gamma_rev = args.gamma_rev
    if args.text_model is not None:
        config.model.text_model_name = args.text_model
    if args.image_model is not None:
        config.model.image_model_name = args.image_model
    if args.shared_dim is not None:
        config.model.shared_dim = args.shared_dim
    if args.max_span_length is not None:
        config.model.max_span_length = args.max_span_length

    # --- Setup ---
    set_seed(config.train.seed + rank)
    os.makedirs(config.train.output_dir, exist_ok=True)
    if is_main:
        setup_logging(os.path.join(config.train.output_dir, "train.log"))
    else:
        logging.basicConfig(level=logging.WARNING)
    device = torch.device(f"cuda:{local_rank}")
    config.train.device = str(device)
    if is_main:
        logger.info(f"Config: {config}")
        logger.info(f"World size: {world_size}")
        if args.select_best_by == "test_f1":
            logger.warning(
                "select_best_by=test_f1 uses the test set for checkpoint selection. "
                "Use this only for diagnostics, not for paper-valid model selection."
            )
        elif args.eval_test_each_epoch:
            logger.warning(
                "eval_test_each_epoch logs test metrics during training. "
                "Keep paper-valid model selection on dev_f1."
            )

    # --- Data ---
    processor = MNERProcessor(
        text_model_name=config.model.text_model_name,
        image_model_name=config.model.image_model_name,
        max_seq_length=config.model.max_seq_length,
        max_span_length=config.model.max_span_length,
    )

    train_dataset = TwitterMNERDataset(
        data_file=os.path.join(config.train.data_dir, args.dataset, "train.txt"),
        image_dir=config.train.image_dir,
        processor=processor,
        is_train=True,
    )
    dev_dataset = TwitterMNERDataset(
        data_file=os.path.join(config.train.data_dir, args.dataset, "dev.txt"),
        image_dir=config.train.image_dir,
        processor=processor,
        is_train=False,
    )
    test_dataset = TwitterMNERDataset(
        data_file=os.path.join(config.train.data_dir, args.dataset, "test.txt"),
        image_dir=config.train.image_dir,
        processor=processor,
        is_train=False,
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
            f"{os.path.join(config.train.data_dir, args.dataset, 'train.txt')}"
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
    )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(raw_model, config.train)
    lr_schedule_epochs = args.lr_schedule_epochs or config.train.epochs
    num_training_steps = (
        len(train_loader) * lr_schedule_epochs
        // config.train.gradient_accumulation_steps
    )
    if is_main and lr_schedule_epochs != config.train.epochs:
        logger.info(
            "LR scheduler uses %d epoch(s) for total steps while training runs %d epoch(s)",
            lr_schedule_epochs,
            config.train.epochs,
        )
    scheduler = build_scheduler(optimizer, num_training_steps, config.train)

    scaler = GradScaler("cuda") if config.train.fp16 else None

    # --- EMA ---
    ema = ModelEMA(raw_model, decay=args.ema_decay)

    # --- Training Loop ---
    selection_metric = args.select_best_by
    eval_test_each_epoch = args.eval_test_each_epoch or selection_metric == "test_f1"
    best_score = -1.0
    best_epoch = 0
    best_dev_metrics = None
    best_test_metrics_at_epoch = None
    best_dev_f1 = 0.0
    best_dev_epoch = 0
    epoch_losses = []
    epoch_metrics = []
    stopped_early = False
    stop_epoch = None
    stop_reason = None
    epochs_ran = 0
    no_improve_epochs = 0
    early_stop_enabled = args.early_stop
    warmup_epochs_ran = args.warmup_imagination_epochs if args.warmup_imagination_epochs > 0 else 0

    try:
        for epoch in range(1, config.train.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main:
                logger.info(f"{'='*60}")
                logger.info(f"Epoch {epoch}/{config.train.epochs}")

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
                    f"L_ira: {train_metrics['l_ira']:.4f} | "
                    f"L_ico: {train_metrics['l_ico']:.4f} | "
                    f"L_sig: {train_metrics['l_sig']:.4f} | "
                    f"L_ira_r: {train_metrics['l_ira_rev']:.4f} | "
                    f"L_ico_r: {train_metrics['l_ico_rev']:.4f} | "
                    f"L_sig_r: {train_metrics['l_sig_rev']:.4f} | "
                    f"Time: {t_elapsed:.1f}s"
                )
                epoch_losses.append(train_metrics['loss'])

                # --- Dev/Test Evaluation with EMA weights (rank 0 only) ---
                ema.apply_shadow(raw_model)
                dev_metrics = evaluate_model(raw_model, dev_loader, config, device)
                test_metrics_for_epoch = None
                if eval_test_each_epoch:
                    test_metrics_for_epoch = evaluate_model(raw_model, test_loader, config, device)
                threshold_metrics_for_epoch = None
                if args.tune_threshold_each_epoch:
                    epoch_thr, tuned_dev_for_epoch = tune_threshold_on_dev(
                        raw_model, dev_loader, config, device,
                    )
                    tuned_test_for_epoch = evaluate_model(
                        raw_model, test_loader, config, device,
                        margin_threshold=epoch_thr,
                    )
                    threshold_metrics_for_epoch = {
                        "margin_threshold": epoch_thr,
                        "tuned_dev": tuned_dev_for_epoch,
                        "tuned_test": tuned_test_for_epoch,
                    }
                ema.restore(raw_model)
                epochs_ran = epoch

                logger.info(
                    f"Dev   | P: {dev_metrics['precision']:.4f} "
                    f"R: {dev_metrics['recall']:.4f} "
                    f"F1: {dev_metrics['f1']:.4f}"
                )
                for etype, ef1 in dev_metrics.get("per_type_f1", {}).items():
                    logger.info(f"  {etype}: F1={ef1:.4f}")

                if test_metrics_for_epoch is not None:
                    logger.info(
                        f"Test@Epoch | P: {test_metrics_for_epoch['precision']:.4f} "
                        f"R: {test_metrics_for_epoch['recall']:.4f} "
                        f"F1: {test_metrics_for_epoch['f1']:.4f}"
                    )
                    for etype, ef1 in test_metrics_for_epoch.get("per_type_f1", {}).items():
                        logger.info(f"  {etype}: F1={ef1:.4f}")

                if threshold_metrics_for_epoch is not None:
                    tuned_dev_for_epoch = threshold_metrics_for_epoch["tuned_dev"]
                    tuned_test_for_epoch = threshold_metrics_for_epoch["tuned_test"]
                    epoch_thr = threshold_metrics_for_epoch["margin_threshold"]
                    logger.info(
                        f"Threshold@Epoch | threshold={epoch_thr:.2f} | "
                        f"Dev P: {tuned_dev_for_epoch['precision']:.4f} "
                        f"R: {tuned_dev_for_epoch['recall']:.4f} "
                        f"F1: {tuned_dev_for_epoch['f1']:.4f}"
                    )
                    logger.info(
                        f"Test@Threshold@Epoch | threshold={epoch_thr:.2f} | "
                        f"P: {tuned_test_for_epoch['precision']:.4f} "
                        f"R: {tuned_test_for_epoch['recall']:.4f} "
                        f"F1: {tuned_test_for_epoch['f1']:.4f}"
                    )
                    for etype, ef1 in tuned_test_for_epoch.get("per_type_f1", {}).items():
                        logger.info(f"  {etype}: F1={ef1:.4f}")

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "dev": dev_metrics,
                }
                if test_metrics_for_epoch is not None:
                    epoch_record["test"] = test_metrics_for_epoch
                if threshold_metrics_for_epoch is not None:
                    epoch_record["threshold"] = threshold_metrics_for_epoch
                epoch_metrics.append(epoch_record)

                had_previous_dev_best = best_dev_epoch > 0
                previous_best_dev_f1 = best_dev_f1
                improvement = dev_metrics["f1"] - previous_best_dev_f1

                if dev_metrics["f1"] > best_dev_f1:
                    best_dev_f1 = dev_metrics["f1"]
                    best_dev_epoch = epoch

                current_score = (
                    test_metrics_for_epoch["f1"]
                    if selection_metric == "test_f1"
                    else dev_metrics["f1"]
                )

                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_dev_metrics = dev_metrics
                    best_test_metrics_at_epoch = test_metrics_for_epoch
                    save_path = os.path.join(config.train.output_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "best_f1": best_score,
                        "best_score": best_score,
                        "selection_metric": selection_metric,
                        "best_dev_f1": dev_metrics["f1"],
                        "best_test_f1": (
                            test_metrics_for_epoch["f1"]
                            if test_metrics_for_epoch is not None
                            else None
                        ),
                        "model_config": asdict(config.model),
                    }, save_path)
                    test_msg = (
                        f", test_f1={test_metrics_for_epoch['f1']:.4f}"
                        if (
                            test_metrics_for_epoch is not None
                            and selection_metric != "test_f1"
                        )
                        else ""
                    )
                    logger.info(
                        f"New best model saved "
                        f"({selection_metric}={best_score:.4f}, "
                        f"dev_f1={dev_metrics['f1']:.4f}{test_msg})"
                    )

                stop_training = False
                if early_stop_enabled:
                    if not had_previous_dev_best:
                        no_improve_epochs = 0
                    elif improvement >= args.early_stop_min_delta:
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1

                    logger.info(
                        "Early stop tracker | metric=%s | current=%.4f | "
                        "best_before_epoch=%.4f | improvement=%.4f | "
                        "patience=%d/%d | min_epochs=%d",
                        args.early_stop_metric,
                        dev_metrics["f1"],
                        previous_best_dev_f1,
                        improvement,
                        no_improve_epochs,
                        args.early_stop_patience,
                        args.early_stop_min_epochs,
                    )

                    if (
                        epoch >= args.early_stop_min_epochs
                        and no_improve_epochs >= args.early_stop_patience
                    ):
                        stopped_early = True
                        stop_epoch = epoch
                        stop_reason = (
                            f"{args.early_stop_metric} improvement "
                            f"({improvement:.4f}) < min_delta "
                            f"({args.early_stop_min_delta}) for "
                            f"{no_improve_epochs} consecutive epoch(s)"
                        )
                        stop_training = True
                        logger.info(
                            "Early stopping triggered at epoch %d: %s",
                            epoch,
                            stop_reason,
                        )
                else:
                    stop_training = False
            else:
                stop_training = False

            if world_size > 1:
                stop_tensor = torch.tensor(
                    [1 if stop_training else 0], device=device, dtype=torch.int
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
                dist.barrier()

            if stop_training:
                break

        if is_main:
            logger.info(f"{'='*60}")
            logger.info(
                f"Best {selection_metric}: {best_score:.4f} at epoch {best_epoch}"
            )
            logger.info(
                f"Best observed dev F1: {best_dev_f1:.4f} at epoch {best_dev_epoch}"
            )

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
            last_ckpt_path = os.path.join(config.train.output_dir, "last_model.pt")
            torch.save({
                "epoch": epochs_ran,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "selection_metric": selection_metric,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_dev_epoch": best_dev_epoch,
                "best_dev_f1": best_dev_f1,
                "model_config": asdict(config.model),
            }, last_ckpt_path)

            final_ckpt_path = (
                last_ckpt_path
                if args.final_eval_checkpoint == "last"
                else os.path.join(config.train.output_dir, "best_model.pt")
            )
            logger.info(
                f"Final evaluation checkpoint: {args.final_eval_checkpoint} "
                f"({final_ckpt_path})"
            )
            best_ckpt = torch.load(final_ckpt_path, map_location=device)
            if "ema_state_dict" in best_ckpt:
                raw_model.load_state_dict(best_ckpt["ema_state_dict"])
            else:
                raw_model.load_state_dict(best_ckpt["model_state_dict"])

            raw_test_metrics = None
            if args.eval_raw_test_final:
                raw_test_metrics = evaluate_model(raw_model, test_loader, config, device)
                logger.info(
                    f"Test@FinalRaw | P: {raw_test_metrics['precision']:.4f} "
                    f"R: {raw_test_metrics['recall']:.4f} "
                    f"F1: {raw_test_metrics['f1']:.4f}"
                )
                for etype, ef1 in raw_test_metrics.get("per_type_f1", {}).items():
                    logger.info(f"  {etype}: F1={ef1:.4f}")

            # Tune margin threshold on dev set, then apply to test
            best_thr, tuned_dev = tune_threshold_on_dev(
                raw_model, dev_loader, config, device,
            )
            logger.info(f"Using tuned margin_threshold={best_thr:.2f} for test")

            test_metrics = evaluate_model(
                raw_model, test_loader, config, device,
                margin_threshold=best_thr,
            )
            logger.info(
                f"Test  | P: {test_metrics['precision']:.4f} "
                f"R: {test_metrics['recall']:.4f} "
                f"F1: {test_metrics['f1']:.4f}"
            )
            for etype, ef1 in test_metrics.get("per_type_f1", {}).items():
                logger.info(f"  {etype}: F1={ef1:.4f}")

            with open(os.path.join(config.train.output_dir, "results.json"), "w") as f:
                json.dump({
                    "status": "completed",
                    "best_epoch": best_epoch,
                    "selection_metric": selection_metric,
                    "best_score": best_score,
                    "best_dev_epoch": best_dev_epoch,
                    "best_dev_f1": best_dev_f1,
                    "best_checkpoint_path": os.path.join(config.train.output_dir, "best_model.pt"),
                    "last_checkpoint_path": last_ckpt_path,
                    "final_eval_checkpoint": args.final_eval_checkpoint,
                    "final_eval_checkpoint_path": final_ckpt_path,
                    "best_margin_threshold": best_thr,
                    "dev": best_dev_metrics,
                    "test_at_best_epoch": best_test_metrics_at_epoch,
                    "raw_test": raw_test_metrics,
                    "tuned_dev": tuned_dev,
                    "test": test_metrics,
                    "epoch_losses": epoch_losses,
                    "epoch_metrics": epoch_metrics,
                    "test_each_epoch_enabled": eval_test_each_epoch,
                    "threshold_each_epoch_enabled": args.tune_threshold_each_epoch,
                    "stopped_early": stopped_early,
                    "stop_epoch": stop_epoch,
                    "stop_reason": stop_reason,
                    "ran_epochs": epochs_ran,
                    "warmup_epochs_ran": warmup_epochs_ran,
                    "early_stop": {
                        "enabled": early_stop_enabled,
                        "metric": args.early_stop_metric,
                        "min_delta": args.early_stop_min_delta,
                        "patience": args.early_stop_patience,
                        "min_epochs": args.early_stop_min_epochs,
                    },
                    "world_size": world_size,
                    "lr_schedule_epochs": lr_schedule_epochs,
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
