"""
ImaginE: Type-Conditioned Imagination World Model for Multimodal NER
Configuration and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # --- Backbone Encoders ---
    text_model_name: str = "cardiffnlp/twitter-roberta-base"
    image_model_name: str = "openai/clip-vit-base-patch16"
    text_hidden_size: int = 768
    image_hidden_size: int = 768

    # --- Shared Projection ---
    shared_dim: int = 384

    # --- Span Encoder ---
    max_span_length: int = 4
    max_seq_length: int = 128

    # --- Entity Type ---
    num_types: int = 5  # PER, LOC, ORG, MISC, O
    type_embed_dim: int = 128

    # --- Imagination Predictor ---
    pred_hidden_dim: int = 512
    pred_num_layers: int = 4
    pred_ffn_expansion: int = 2
    pred_dropout: float = 0.1

    # --- Comparator ---
    comparator_hidden_dim: int = 256

    # --- Reverse Imagination (Bidirectional) ---
    rev_pred_hidden_dim: int = 512
    rev_pred_num_layers: int = 4
    rev_pred_ffn_expansion: int = 2
    rev_pred_dropout: float = 0.1
    rev_comparator_hidden_dim: int = 256
    share_type_embedding: bool = True

    # --- Classifier ---
    classifier_hidden_dim: int = 256
    width_embed_dim: int = 32


@dataclass
class LossConfig:
    alpha: float = 1.0    # L_ira weight (forward)
    beta: float = 0.5     # L_ico weight (forward)
    gamma: float = 0.05   # L_sig weight (forward)
    tau: float = 0.07     # temperature for L_ico
    label_smoothing: float = 0.02  # label smoothing for CE loss
    r_drop_alpha: float = 0.5  # R-Drop KL divergence weight (0 = disabled)

    # --- Reverse imagination loss weights ---
    alpha_rev: float = 1.0   # L_ira_rev weight (reverse)
    beta_rev: float = 0.5    # L_ico_rev weight (reverse)
    gamma_rev: float = 0.05  # L_sig_rev weight (reverse)


@dataclass
class TrainConfig:
    # --- Data ---
    dataset: str = "twitter2017"  # twitter2015 | twitter2017
    data_dir: str = "./data"
    image_dir: str = "./data/images"

    # --- Training ---
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # --- Optimizer ---
    encoder_lr: float = 2e-5
    new_module_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    # --- Evaluation ---
    save_best: bool = True
    metric_for_best: str = "f1"

    # --- Paths ---
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    # --- Device ---
    device: str = "cuda"
    fp16: bool = True
    num_workers: int = 2


@dataclass
class ImaginEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


ENTITY_TYPES = ["O", "PER", "LOC", "ORG", "MISC"]
ENTITY_TYPE_TO_ID = {t: i for i, t in enumerate(ENTITY_TYPES)}
# Twitter-2015 uses "OTHER" instead of "MISC"; map it to the same ID
ENTITY_TYPE_TO_ID["OTHER"] = ENTITY_TYPE_TO_ID["MISC"]
ID_TO_ENTITY_TYPE = {i: t for i, t in enumerate(ENTITY_TYPES)}
