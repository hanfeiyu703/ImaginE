"""
ImaginE composite loss function (Bidirectional Imagination).

Total loss = L_task
           + α·L_ira     + α_rev·L_ira_rev
           + β·L_ico     + β_rev·L_ico_rev
           + γ·L_sig     + γ_rev·L_sig_rev
           + λ_groundable·L_groundable
           + λ_reg·L_register + λ_qsp·L_qsp
           + λ_coarse·L_coarse + λ_cf·L_coarse_fine
           + λ_noreg·L_no_region_consistency
           + λ_patch·L_clip_patch

Forward losses (text imagines image):
    L_task: Span-level cross-entropy loss (standard NER)
    L_ira:  Imagination-Reality Alignment loss (MSE with stop-gradient)
    L_ico:  Imagination Contrastive loss (InfoNCE over type scores)
    L_sig:  SIGReg anti-collapse regularization

Reverse losses (image imagines text):
    L_ira_rev: Reverse Alignment loss (MSE between imagined text and real span)
    L_ico_rev: Reverse Contrastive loss (InfoNCE over reverse scores)
    L_sig_rev: Reverse SIGReg on imagined text representations

Dream register losses:
    L_groundable: H-Index-style binary grounding existence supervision
    L_register: batch-level InfoNCE between visual register summary and entity text summary
    L_qsp: lightweight query similarity preservation over entity span features

FMNERG coarse-fine losses:
    L_coarse: auxiliary coarse entity classification
    L_coarse_fine: consistency between coarse logits and fine logits aggregated by tree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:  # pragma: no cover - exercised only without scipy
    linear_sum_assignment = None


class ImaginELoss(nn.Module):
    """Combined loss for ImaginE training with bidirectional imagination."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        tau: float = 0.07,
        label_smoothing: float = 0.1,
        num_types: int = 5,
        alpha_rev: float = 1.0,
        beta_rev: float = 0.5,
        gamma_rev: float = 0.05,
        grounding_weight: float = 1.0,
        groundable_weight: float = 0.0,
        register_weight: float = 0.1,
        qsp_weight: float = 0.05,
        num_coarse_types: int | None = None,
        fine_to_coarse_ids: list[int] | None = None,
        coarse_weight: float = 0.0,
        coarse_fine_weight: float = 0.0,
        no_region_consistency_weight: float = 0.0,
        clip_patch_weight: float = 0.0,
        region_hard_negative_weight: float = 0.0,
        fine_loss_type: str = "ce",
        fine_focal_gamma: float = 1.5,
        fine_class_balance_beta: float = 0.999,
        fine_class_counts: list[int] | None = None,
        set_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_rev = alpha_rev
        self.beta_rev = beta_rev
        self.gamma_rev = gamma_rev
        self.tau = tau
        self.num_types = num_types
        self.grounding_weight = grounding_weight
        self.groundable_weight = groundable_weight
        self.register_weight = register_weight
        self.qsp_weight = qsp_weight
        self.num_coarse_types = num_coarse_types or num_types
        self.coarse_weight = coarse_weight
        self.coarse_fine_weight = coarse_fine_weight
        self.no_region_consistency_weight = no_region_consistency_weight
        self.clip_patch_weight = clip_patch_weight
        self.region_hard_negative_weight = region_hard_negative_weight
        self.fine_loss_type = fine_loss_type
        self.fine_focal_gamma = fine_focal_gamma
        self.fine_class_balance_beta = fine_class_balance_beta
        self.set_aux_weight = set_aux_weight

        class_weights = torch.tensor(
            [0.25] + [1.0] * (num_types - 1), dtype=torch.float
        )
        self.register_buffer("class_weights", class_weights)
        if fine_class_counts is None or len(fine_class_counts) != num_types:
            fine_class_counts = [1] * num_types
        cb_weights = self._class_balanced_weights(
            fine_class_counts,
            beta=fine_class_balance_beta,
        )
        self.register_buffer("fine_cb_weights", cb_weights)
        coarse_class_weights = torch.tensor(
            [0.25] + [1.0] * (self.num_coarse_types - 1), dtype=torch.float
        )
        self.register_buffer("coarse_class_weights", coarse_class_weights)
        if fine_to_coarse_ids is None:
            if self.num_coarse_types == num_types:
                fine_to_coarse_ids = list(range(num_types))
            else:
                fine_to_coarse_ids = [0] * num_types
        if len(fine_to_coarse_ids) != num_types:
            raise ValueError("fine_to_coarse_ids length must match num_types.")
        self.register_buffer(
            "fine_to_coarse_ids",
            torch.tensor(fine_to_coarse_ids, dtype=torch.long),
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean",
            label_smoothing=label_smoothing,
        )
        self.label_smoothing = label_smoothing
        self.ico_ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.sigreg = SIGReg()

    @staticmethod
    def _class_balanced_weights(counts: list[int], beta: float) -> torch.Tensor:
        counts_tensor = torch.tensor(counts, dtype=torch.float).clamp(min=1.0)
        if beta <= 0.0 or beta >= 1.0:
            weights = torch.ones_like(counts_tensor)
        else:
            effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float), counts_tensor)
            weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
            weights = weights / weights.mean().clamp(min=1e-8)
        if weights.numel() > 0:
            weights[0] = 0.25
        return weights

    @staticmethod
    def _zero_like_loss(anchor: torch.Tensor) -> torch.Tensor:
        """Return a differentiable scalar zero on the anchor's device/dtype."""
        return anchor.sum() * 0.0

    def task_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Span-level cross-entropy loss with label smoothing.

        Args:
            logits: (B, S, K) classification logits
            labels: (B, S) ground-truth type indices
            mask:   (B, S) boolean mask for valid spans
        Returns:
            scalar loss
        """
        valid = mask.bool()
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        flat_logits = logits[valid]  # (N_valid, K)
        flat_labels = labels[valid]  # (N_valid,)

        if self.ce_loss.weight.device != flat_logits.device:
            self.ce_loss.weight = self.class_weights.to(flat_logits.device)

        if self.fine_loss_type == "cb_focal":
            cb_weights = self.fine_cb_weights.to(flat_logits.device)
            ce = F.cross_entropy(
                flat_logits.float(),
                flat_labels,
                weight=cb_weights,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            probs = F.softmax(flat_logits.float(), dim=-1)
            pt = probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)
            focal = torch.pow((1.0 - pt).clamp(min=0.0), self.fine_focal_gamma)
            return (focal * ce).mean()

        return self.ce_loss(flat_logits, flat_labels)

    def imagination_reality_alignment_loss(
        self,
        z_imag: torch.Tensor,
        z_attended: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """L_ira: MSE between imagined visual and real visual for the GT type.

        Only the correct type's imagination is aligned to reality.
        Stop-gradient on the visual side to prevent collapse.
        Only computed on entity spans (label != 0).

        Args:
            z_imag:     (B, S, K, d) imagined visual representations
            z_attended: (B, S, K, d) attention-weighted real visual features
            labels:     (B, S) ground-truth type indices
            mask:       (B, S) valid span mask
        Returns:
            scalar loss
        """
        entity_mask = mask.bool() & (labels != 0)
        if not entity_mask.any():
            return torch.tensor(0.0, device=z_imag.device, requires_grad=True)

        B, S, K, d = z_imag.shape

        gt_idx = labels.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, d)

        z_imag_gt = torch.gather(z_imag, 2, gt_idx).squeeze(2)          # (B, S, d)
        z_attend_gt = torch.gather(z_attended, 2, gt_idx).squeeze(2)     # (B, S, d)

        z_attend_gt = z_attend_gt.detach()

        diff = z_imag_gt[entity_mask] - z_attend_gt[entity_mask]  # (N_entity, d)
        return (diff ** 2).mean()

    def reverse_imagination_reality_alignment_loss(
        self,
        z_imag_text: torch.Tensor,
        z_span: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        visual_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L_ira_rev: MSE between imagined text and real text span for the GT type.

        Stop-gradient on the text span side (let imagination adapt to reality).
        Only computed on entity spans (label != 0).
        When visual_gate is provided, each sample's loss is weighted by
        the gate value so irrelevant images contribute less.

        Args:
            z_imag_text: (B, S, K, d) imagined textual representations
            z_span:      (B, S, d) real text span representations
            labels:      (B, S) ground-truth type indices
            mask:        (B, S) valid span mask
            visual_gate: (B, S) per-span relevance weights ∈ [0, 1], optional
        Returns:
            scalar loss
        """
        entity_mask = mask.bool() & (labels != 0)
        if not entity_mask.any():
            return torch.tensor(0.0, device=z_imag_text.device, requires_grad=True)

        B, S, K, d = z_imag_text.shape

        gt_idx = labels.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, d)
        z_imag_text_gt = torch.gather(z_imag_text, 2, gt_idx).squeeze(2)  # (B, S, d)

        z_span_target = z_span.detach()

        diff = z_imag_text_gt[entity_mask] - z_span_target[entity_mask]  # (N_entity, d)
        per_sample = (diff ** 2).mean(dim=-1)  # (N_entity,)

        if visual_gate is not None:
            weights = visual_gate[entity_mask].detach()  # (N_entity,)
            return (per_sample * weights).mean()
        return per_sample.mean()

    def imagination_contrastive_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L_ico: InfoNCE — correct type's score should be highest.

        L_ico = -log( exp(s_y/tau) / sum_k exp(s_k/tau) )

        Only computed on entity spans (label != 0).
        When sample_weights is provided, each sample's CE is weighted
        so irrelevant images contribute less.

        Args:
            scores:         (B, S, K) imagination-reality compatibility scores
            labels:         (B, S) ground-truth type indices
            mask:           (B, S) valid span mask
            sample_weights: (B, S) per-span weights ∈ [0, 1], optional
        Returns:
            scalar loss
        """
        entity_mask = mask.bool() & (labels != 0)
        if not entity_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        flat_scores = scores[entity_mask].float() / self.tau  # (N_entity, K)
        flat_labels = labels[entity_mask]                      # (N_entity,)

        if sample_weights is not None:
            weights = sample_weights[entity_mask].detach()  # (N_entity,)
            per_sample_ce = F.cross_entropy(flat_scores, flat_labels, reduction="none")
            return (per_sample_ce * weights).mean()
        return self.ico_ce_loss(flat_scores, flat_labels)

    def sigreg_loss(
        self, z_imag: torch.Tensor, span_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_sig: SIGReg on entity-span imagined representations to prevent collapse.

        Only entity spans (label != 0) are included.

        Args:
            z_imag:    (B, S, K, d) imagined representations
            span_mask: (B, S) valid span mask (1=valid, 0=padding)
            labels:    (B, S) ground-truth type indices
        Returns:
            scalar loss
        """
        B, S, K, d = z_imag.shape
        valid = span_mask.bool() & (labels != 0)
        valid_expanded = valid.unsqueeze(-1).unsqueeze(-1).expand(B, S, K, d)
        z_flat = z_imag[valid_expanded].reshape(-1, d)  # (N_valid*K, d)

        if z_flat.size(0) < 4:
            return torch.tensor(0.0, device=z_imag.device, requires_grad=True)

        max_samples = 2048
        if z_flat.size(0) > max_samples:
            indices = torch.randperm(z_flat.size(0), device=z_flat.device)[:max_samples]
            z_flat = z_flat[indices]

        return self.sigreg(z_flat)

    def grounding_loss(
        self,
        grounding_logits: torch.Tensor,
        grounding_labels: torch.Tensor,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
        groundable_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Soft-target grounding loss on entity spans only."""
        entity_mask = span_mask.bool() & (labels != 0)
        if groundable_labels is not None:
            region_mass = grounding_labels[..., :-1].sum(dim=-1)
            detector_miss = (groundable_labels > 0.5) & (region_mass <= 0)
            entity_mask = entity_mask & ~detector_miss
        if not entity_mask.any():
            return torch.tensor(0.0, device=grounding_logits.device, requires_grad=True)

        log_probs = F.log_softmax(grounding_logits[entity_mask], dim=-1)
        targets = grounding_labels[entity_mask]
        return F.kl_div(log_probs, targets, reduction="batchmean")

    def groundable_loss(
        self,
        groundable_logits: torch.Tensor | None,
        groundable_labels: torch.Tensor | None,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Binary supervision for whether an entity should be grounded."""
        if groundable_logits is None or groundable_labels is None:
            return self._zero_like_loss(labels.to(span_mask.dtype))

        entity_mask = span_mask.bool() & (labels != 0)
        if not entity_mask.any():
            return self._zero_like_loss(groundable_logits)

        return F.binary_cross_entropy_with_logits(
            groundable_logits[entity_mask].float(),
            groundable_labels[entity_mask].float(),
        )

    def region_hard_negative_loss(
        self,
        grounding_logits: torch.Tensor | None,
        grounding_labels: torch.Tensor | None,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Margin loss against high-scoring non-overlapping region proposals."""
        if grounding_logits is None or grounding_labels is None:
            return self._zero_like_loss(labels.to(span_mask.dtype))
        if grounding_logits.size(-1) < 2:
            return self._zero_like_loss(grounding_logits)

        entity_mask = span_mask.bool() & (labels != 0)
        region_targets = grounding_labels[..., :-1].float()
        positive_region_mask = region_targets > 0
        usable = entity_mask & positive_region_mask.any(dim=-1)
        if not usable.any():
            return self._zero_like_loss(grounding_logits)

        region_logits = grounding_logits[..., :-1].float()[usable]
        positives = positive_region_mask[usable]
        negatives = ~positives
        if not negatives.any():
            return self._zero_like_loss(grounding_logits)

        positive_score = torch.logsumexp(
            region_logits.masked_fill(~positives, -1e4),
            dim=-1,
        )
        negative_score = torch.logsumexp(
            region_logits.masked_fill(~negatives, -1e4),
            dim=-1,
        )
        return F.relu(margin + negative_score - positive_score).mean()

    def no_region_consistency_loss(
        self,
        grounding_logits: torch.Tensor | None,
        groundable_logits: torch.Tensor | None,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Align groundable logits with the grounding head's region-vs-none belief."""
        if grounding_logits is None or groundable_logits is None:
            return self._zero_like_loss(labels.to(span_mask.dtype))

        entity_mask = span_mask.bool() & (labels != 0)
        if not entity_mask.any():
            return self._zero_like_loss(groundable_logits)
        if grounding_logits.size(-1) < 2:
            return self._zero_like_loss(grounding_logits)

        selected_grounding = grounding_logits[entity_mask].float()
        selected_groundable = groundable_logits[entity_mask].float()

        region_logit = torch.logsumexp(selected_grounding[:, :-1], dim=-1)
        no_region_logit = selected_grounding[:, -1]
        pointer_groundable_logit = region_logit - no_region_logit

        pointer_target = torch.sigmoid(pointer_groundable_logit.detach())
        groundable_target = torch.sigmoid(selected_groundable.detach())
        loss_groundable = F.binary_cross_entropy_with_logits(
            selected_groundable,
            pointer_target,
        )
        loss_pointer = F.binary_cross_entropy_with_logits(
            pointer_groundable_logit,
            groundable_target,
        )
        return 0.5 * (loss_groundable + loss_pointer)

    def clip_patch_grounding_loss(
        self,
        clip_patch_logits: torch.Tensor | None,
        clip_patch_labels: torch.Tensor | None,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft grounding supervision over frozen-CLIP patch tokens."""
        if clip_patch_logits is None or clip_patch_labels is None:
            return self._zero_like_loss(labels.to(span_mask.dtype))
        if clip_patch_logits.shape[:-1] != clip_patch_labels.shape[:-1]:
            return self._zero_like_loss(clip_patch_logits)

        common_patches = min(clip_patch_logits.size(-1), clip_patch_labels.size(-1))
        if common_patches <= 0:
            return self._zero_like_loss(clip_patch_logits)
        if clip_patch_logits.size(-1) != clip_patch_labels.size(-1):
            clip_patch_logits = clip_patch_logits[..., :common_patches]
            clip_patch_labels = clip_patch_labels[..., :common_patches]

        entity_mask = span_mask.bool() & (labels != 0)
        label_mass = clip_patch_labels.sum(dim=-1) > 0
        patch_mask = entity_mask & label_mass
        if not patch_mask.any():
            return self._zero_like_loss(clip_patch_logits)

        log_probs = F.log_softmax(clip_patch_logits[patch_mask].float(), dim=-1)
        targets = clip_patch_labels[patch_mask].float()
        return F.kl_div(log_probs, targets, reduction="batchmean")

    @staticmethod
    def _greedy_match(cost: torch.Tensor) -> tuple[list[int], list[int]]:
        rows: list[int] = []
        cols: list[int] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat_order = torch.argsort(cost.reshape(-1))
        num_rows, num_cols = cost.shape
        for flat_idx in flat_order.tolist():
            row = flat_idx // num_cols
            col = flat_idx % num_cols
            if row in used_rows or col in used_cols:
                continue
            rows.append(row)
            cols.append(col)
            used_rows.add(row)
            used_cols.add(col)
            if len(rows) >= min(num_rows, num_cols):
                break
        return rows, cols

    def _match_set_queries(
        self,
        cost: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        if cost.numel() <= 0:
            return [], []
        cpu_cost = cost.detach().float().cpu()
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cpu_cost.numpy())
            return list(row_ind), list(col_ind)
        return self._greedy_match(cpu_cost)

    def set_prediction_aux_loss(
        self,
        set_aux_outputs: dict[str, torch.Tensor] | None,
        labels: torch.Tensor,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
        grounding_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fixed-query auxiliary set prediction loss for entity recall."""
        if set_aux_outputs is None:
            return self._zero_like_loss(labels.to(span_mask.dtype))

        start_logits = set_aux_outputs["start_logits"]
        end_logits = set_aux_outputs["end_logits"]
        type_logits = set_aux_outputs["type_logits"]
        set_grounding_logits = set_aux_outputs.get("grounding_logits")

        batch_size, num_queries, _ = type_logits.shape
        total_losses = []
        for batch_idx in range(batch_size):
            entity_mask = span_mask[batch_idx].bool() & (labels[batch_idx] != 0)
            gold_labels = labels[batch_idx][entity_mask].clamp(
                0,
                type_logits.size(-1) - 1,
            )
            gold_spans = span_indices[batch_idx][entity_mask]
            if gold_labels.numel() == 0:
                no_object = F.cross_entropy(
                    type_logits[batch_idx].float(),
                    torch.zeros(num_queries, dtype=torch.long, device=type_logits.device),
                    reduction="mean",
                )
                total_losses.append(0.2 * no_object)
                continue

            gold_starts = gold_spans[:, 0].clamp(0, start_logits.size(-1) - 1)
            gold_ends = gold_spans[:, 1].clamp(0, end_logits.size(-1) - 1)

            with torch.no_grad():
                type_cost = -F.log_softmax(type_logits[batch_idx].float(), dim=-1)[
                    :,
                    gold_labels,
                ]
                start_cost = -F.log_softmax(start_logits[batch_idx].float(), dim=-1)[
                    :,
                    gold_starts,
                ]
                end_cost = -F.log_softmax(end_logits[batch_idx].float(), dim=-1)[
                    :,
                    gold_ends,
                ]
                match_cost = type_cost + 0.5 * (start_cost + end_cost)

            query_ids, gold_ids = self._match_set_queries(match_cost)
            if not query_ids:
                continue
            query_tensor = torch.tensor(query_ids, dtype=torch.long, device=type_logits.device)
            gold_tensor = torch.tensor(gold_ids, dtype=torch.long, device=type_logits.device)

            matched_type = F.cross_entropy(
                type_logits[batch_idx, query_tensor].float(),
                gold_labels[gold_tensor],
                reduction="mean",
            )
            matched_start = F.cross_entropy(
                start_logits[batch_idx, query_tensor].float(),
                gold_starts[gold_tensor],
                reduction="mean",
            )
            matched_end = F.cross_entropy(
                end_logits[batch_idx, query_tensor].float(),
                gold_ends[gold_tensor],
                reduction="mean",
            )
            matched_loss = matched_type + 0.5 * (matched_start + matched_end)

            if (
                set_grounding_logits is not None
                and grounding_labels is not None
                and set_grounding_logits.size(-1) == grounding_labels.size(-1)
            ):
                gold_grounding = grounding_labels[batch_idx][entity_mask][gold_tensor].float()
                log_probs = F.log_softmax(
                    set_grounding_logits[batch_idx, query_tensor].float(),
                    dim=-1,
                )
                matched_loss = matched_loss + 0.2 * F.kl_div(
                    log_probs,
                    gold_grounding,
                    reduction="batchmean",
                )

            unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=type_logits.device)
            unmatched_mask[query_tensor] = False
            if unmatched_mask.any():
                no_object = F.cross_entropy(
                    type_logits[batch_idx, unmatched_mask].float(),
                    torch.zeros(
                        int(unmatched_mask.sum().item()),
                        dtype=torch.long,
                        device=type_logits.device,
                    ),
                    reduction="mean",
                )
                matched_loss = matched_loss + 0.2 * no_object
            total_losses.append(matched_loss)

        if not total_losses:
            return self._zero_like_loss(type_logits)
        return torch.stack(total_losses).mean()

    def register_alignment_loss(
        self,
        register_summary: torch.Tensor | None,
        z_span: torch.Tensor,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Align visual register summaries to detached entity-span text summaries."""
        if register_summary is None:
            return self._zero_like_loss(z_span)

        entity_mask = span_mask.bool() & (labels != 0)
        entity_counts = entity_mask.sum(dim=1)
        valid_registers = register_summary.detach().abs().sum(dim=-1) > 0
        valid_samples = (entity_counts > 0) & valid_registers
        if not valid_samples.any():
            return self._zero_like_loss(register_summary)

        weights = entity_mask.to(z_span.dtype).unsqueeze(-1)
        text_summary = (z_span.detach() * weights).sum(dim=1)
        text_summary = text_summary / entity_counts.clamp(min=1).to(z_span.dtype).unsqueeze(-1)

        reg = F.normalize(register_summary[valid_samples], dim=-1)
        text = F.normalize(text_summary[valid_samples], dim=-1)
        logits = torch.matmul(reg, text.transpose(0, 1)) / self.tau
        targets = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.transpose(0, 1), targets)
        return (loss_i2t + loss_t2i) / 2

    def query_similarity_preservation_loss(
        self,
        z_span: torch.Tensor,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Lightweight QSP: same-sample entity spans are positives, others negatives."""
        entity_mask = span_mask.bool() & (labels != 0)
        if not entity_mask.any():
            return self._zero_like_loss(z_span)

        features = z_span[entity_mask]
        if features.size(0) < 2:
            return self._zero_like_loss(z_span)

        batch_size, num_spans, _ = z_span.shape
        sample_ids = torch.arange(batch_size, device=z_span.device)
        sample_ids = sample_ids.unsqueeze(1).expand(batch_size, num_spans)[entity_mask]

        same_sample = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)
        eye = torch.eye(features.size(0), dtype=torch.bool, device=z_span.device)
        positive_mask = same_sample & ~eye
        rows_with_positive = positive_mask.any(dim=1)
        if not rows_with_positive.any():
            return self._zero_like_loss(z_span)

        features = F.normalize(features, dim=-1)
        logits = torch.matmul(features, features.transpose(0, 1)) / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        non_self = (~eye).to(logits.dtype)
        exp_logits = torch.exp(logits) * non_self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

        pos_counts = positive_mask.sum(dim=1).clamp(min=1).to(logits.dtype)
        mean_log_prob_pos = (positive_mask.to(logits.dtype) * log_prob).sum(dim=1) / pos_counts
        return -mean_log_prob_pos[rows_with_positive].mean()

    def coarse_task_loss(
        self,
        coarse_logits: torch.Tensor | None,
        coarse_labels: torch.Tensor | None,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary coarse-type cross entropy on valid spans."""
        if coarse_logits is None or coarse_labels is None:
            return self._zero_like_loss(span_mask)

        valid = span_mask.bool()
        if not valid.any():
            return self._zero_like_loss(coarse_logits)

        return F.cross_entropy(
            coarse_logits[valid].float(),
            coarse_labels[valid].clamp(0, coarse_logits.size(-1) - 1),
            weight=self.coarse_class_weights.to(coarse_logits.device),
            label_smoothing=self.label_smoothing,
        )

    def coarse_fine_consistency_loss(
        self,
        logits: torch.Tensor,
        coarse_logits: torch.Tensor | None,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match coarse logits with fine logits aggregated through the label tree."""
        if coarse_logits is None:
            return self._zero_like_loss(logits)
        if self.fine_to_coarse_ids.numel() != logits.size(-1):
            return self._zero_like_loss(logits)

        entity_mask = span_mask.bool() & (labels != 0)
        if not entity_mask.any():
            return self._zero_like_loss(logits)

        fine_probs = F.softmax(logits[entity_mask].float(), dim=-1)
        fine_to_coarse = self.fine_to_coarse_ids.to(logits.device)
        if fine_to_coarse.max().item() >= coarse_logits.size(-1):
            return self._zero_like_loss(logits)
        coarse_from_fine = torch.zeros(
            fine_probs.size(0),
            coarse_logits.size(-1),
            dtype=fine_probs.dtype,
            device=logits.device,
        )
        coarse_from_fine.scatter_add_(
            dim=1,
            index=fine_to_coarse.unsqueeze(0).expand(fine_probs.size(0), -1),
            src=fine_probs,
        )
        coarse_from_fine = coarse_from_fine.clamp(min=1e-8)

        fine_coarse_log_probs = coarse_from_fine.log()
        coarse_probs = F.softmax(coarse_logits[entity_mask].float(), dim=-1).detach()
        return F.kl_div(
            fine_coarse_log_probs,
            coarse_probs,
            reduction="batchmean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        z_imag: torch.Tensor,
        z_attended: torch.Tensor,
        scores: torch.Tensor,
        z_imag_text: torch.Tensor,
        reverse_scores: torch.Tensor,
        z_span: torch.Tensor,
        labels: torch.Tensor,
        span_mask: torch.Tensor,
        grounding_logits: torch.Tensor | None = None,
        grounding_labels: torch.Tensor | None = None,
        groundable_logits: torch.Tensor | None = None,
        groundable_labels: torch.Tensor | None = None,
        coarse_logits: torch.Tensor | None = None,
        coarse_labels: torch.Tensor | None = None,
        register_summary: torch.Tensor | None = None,
        clip_patch_logits: torch.Tensor | None = None,
        clip_patch_labels: torch.Tensor | None = None,
        set_aux_outputs: dict[str, torch.Tensor] | None = None,
        task: str = "mner",
        visual_gate: torch.Tensor | None = None,
        span_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the composite ImaginE loss (bidirectional).

        Args:
            logits:          (B, S, K) classification logits
            z_imag:          (B, S, K, d) imagined visual representations
            z_attended:      (B, S, K, d) attended real visual features
            scores:          (B, S, K) forward imagination-reality scores
            z_imag_text:     (B, S, K, d) imagined textual representations
            reverse_scores:  (B, S, K) reverse imagination-reality scores
            z_span:          (B, S, d) real text span representations
            labels:          (B, S) ground-truth type ids
            span_mask:       (B, S) valid span mask (1=valid, 0=padding)
            coarse_logits:   (B, S, C) coarse type logits, optional.
            coarse_labels:   (B, S) ground-truth coarse type ids, optional.
            register_summary:(B, d) visual register summary, optional.
            visual_gate:     (B, S) per-span visual relevance ∈ [0, 1], optional.
                             When provided, reverse losses are weighted per-span.
        Returns:
            dict with total loss and individual components
        """
        # --- Forward imagination losses ---
        l_task = self.task_loss(logits, labels, span_mask)
        l_ira = self.imagination_reality_alignment_loss(
            z_imag, z_attended, labels, span_mask
        )
        l_ico = self.imagination_contrastive_loss(scores, labels, span_mask)
        l_sig = self.sigreg_loss(z_imag, span_mask, labels)

        # --- Reverse imagination losses (weighted by visual relevance gate) ---
        l_ira_rev = self.reverse_imagination_reality_alignment_loss(
            z_imag_text, z_span, labels, span_mask,
            visual_gate=visual_gate,
        )
        l_ico_rev = self.imagination_contrastive_loss(
            reverse_scores, labels, span_mask,
            sample_weights=visual_gate,
        )
        l_sig_rev = self.sigreg_loss(z_imag_text, span_mask, labels)
        if task in {"gmner", "fmnerg"} and grounding_logits is not None and grounding_labels is not None:
            l_ground = self.grounding_loss(
                grounding_logits,
                grounding_labels,
                labels,
                span_mask,
                groundable_labels=groundable_labels,
            )
        else:
            l_ground = torch.tensor(0.0, device=logits.device, requires_grad=True)
        if task in {"gmner", "fmnerg"}:
            l_groundable = self.groundable_loss(
                groundable_logits,
                groundable_labels,
                labels,
                span_mask,
            )
            l_register = self.register_alignment_loss(
                register_summary,
                z_span,
                labels,
                span_mask,
            )
            l_qsp = self.query_similarity_preservation_loss(
                z_span,
                labels,
                span_mask,
            )
            l_no_region_consistency = self.no_region_consistency_loss(
                grounding_logits,
                groundable_logits,
                labels,
                span_mask,
            )
            l_clip_patch = self.clip_patch_grounding_loss(
                clip_patch_logits,
                clip_patch_labels,
                labels,
                span_mask,
            )
            l_region_hard_neg = self.region_hard_negative_loss(
                grounding_logits,
                grounding_labels,
                labels,
                span_mask,
            )
        else:
            l_groundable = self._zero_like_loss(logits)
            l_register = self._zero_like_loss(logits)
            l_qsp = self._zero_like_loss(logits)
            l_no_region_consistency = self._zero_like_loss(logits)
            l_clip_patch = self._zero_like_loss(logits)
            l_region_hard_neg = self._zero_like_loss(logits)

        if task == "fmnerg":
            l_coarse = self.coarse_task_loss(
                coarse_logits,
                coarse_labels,
                span_mask,
            )
            l_coarse_fine = self.coarse_fine_consistency_loss(
                logits,
                coarse_logits,
                labels,
                span_mask,
            )
        else:
            l_coarse = self._zero_like_loss(logits)
            l_coarse_fine = self._zero_like_loss(logits)

        if (
            task in {"gmner", "fmnerg"}
            and set_aux_outputs is not None
            and span_indices is not None
        ):
            l_set_aux = self.set_prediction_aux_loss(
                set_aux_outputs,
                labels,
                span_indices,
                span_mask,
                grounding_labels=grounding_labels,
            )
        else:
            l_set_aux = self._zero_like_loss(logits)

        total = (
            l_task
            + self.alpha * l_ira + self.alpha_rev * l_ira_rev
            + self.beta * l_ico + self.beta_rev * l_ico_rev
            + self.gamma * l_sig + self.gamma_rev * l_sig_rev
            + self.grounding_weight * l_ground
            + self.groundable_weight * l_groundable
            + self.register_weight * l_register
            + self.qsp_weight * l_qsp
            + self.coarse_weight * l_coarse
            + self.coarse_fine_weight * l_coarse_fine
            + self.no_region_consistency_weight * l_no_region_consistency
            + self.clip_patch_weight * l_clip_patch
            + self.region_hard_negative_weight * l_region_hard_neg
            + self.set_aux_weight * l_set_aux
        )

        return {
            "loss": total,
            "l_task": l_task.detach(),
            "l_ground": l_ground.detach(),
            "l_groundable": l_groundable.detach(),
            "l_ira": l_ira.detach(),
            "l_ico": l_ico.detach(),
            "l_sig": l_sig.detach(),
            "l_ira_rev": l_ira_rev.detach(),
            "l_ico_rev": l_ico_rev.detach(),
            "l_sig_rev": l_sig_rev.detach(),
            "l_register": l_register.detach(),
            "l_qsp": l_qsp.detach(),
            "l_coarse": l_coarse.detach(),
            "l_coarse_fine": l_coarse_fine.detach(),
            "l_no_region_consistency": l_no_region_consistency.detach(),
            "l_clip_patch": l_clip_patch.detach(),
            "l_region_hard_neg": l_region_hard_neg.detach(),
            "l_set_aux": l_set_aux.detach(),
        }
