"""
ImaginE composite loss function (Bidirectional Imagination).

Total loss = L_task
           + α·L_ira     + α_rev·L_ira_rev
           + β·L_ico     + β_rev·L_ico_rev
           + γ·L_sig     + γ_rev·L_sig_rev

Forward losses (text imagines image):
    L_task: Span-level cross-entropy loss (standard NER)
    L_ira:  Imagination-Reality Alignment loss (MSE with stop-gradient)
    L_ico:  Imagination Contrastive loss (InfoNCE over type scores)
    L_sig:  SIGReg anti-collapse regularization

Reverse losses (image imagines text):
    L_ira_rev: Reverse Alignment loss (MSE between imagined text and real span)
    L_ico_rev: Reverse Contrastive loss (InfoNCE over reverse scores)
    L_sig_rev: Reverse SIGReg on imagined text representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg


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

        class_weights = torch.tensor(
            [0.25] + [1.0] * (num_types - 1), dtype=torch.float
        )
        self.register_buffer("class_weights", class_weights)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean",
            label_smoothing=label_smoothing,
        )
        self.ico_ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.sigreg = SIGReg()

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
        visual_gate: torch.Tensor | None = None,
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

        total = (
            l_task
            + self.alpha * l_ira + self.alpha_rev * l_ira_rev
            + self.beta * l_ico + self.beta_rev * l_ico_rev
            + self.gamma * l_sig + self.gamma_rev * l_sig_rev
        )

        return {
            "loss": total,
            "l_task": l_task.detach(),
            "l_ira": l_ira.detach(),
            "l_ico": l_ico.detach(),
            "l_sig": l_sig.detach(),
            "l_ira_rev": l_ira_rev.detach(),
            "l_ico_rev": l_ico_rev.detach(),
            "l_sig_rev": l_sig_rev.detach(),
        }
