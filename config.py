"""
ImaginE configuration, task definitions, and label spaces.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


TASK_CHOICES = ("mner", "gmner", "fmnerg")
VISUAL_BACKENDS = ("raw_image", "vinvl")


COARSE_ENTITY_TYPES = ["O", "PER", "LOC", "ORG", "MISC"]
COARSE_ENTITY_TYPE_TO_ID = {t: i for i, t in enumerate(COARSE_ENTITY_TYPES)}
# Twitter-2015/2017 and GMNER references sometimes use OTHER instead of MISC.
COARSE_ENTITY_TYPE_TO_ID["OTHER"] = COARSE_ENTITY_TYPE_TO_ID["MISC"]
COARSE_ID_TO_ENTITY_TYPE = {i: t for i, t in enumerate(COARSE_ENTITY_TYPES)}


FMNERG_COARSE_FINE_TREE = {
    "location": [
        "city",
        "country",
        "state",
        "continent",
        "location_other",
        "park",
        "road",
    ],
    "building": [
        "building_other",
        "cultural_place",
        "entertainment_place",
        "sports_facility",
    ],
    "organization": [
        "company",
        "educational_institution",
        "band",
        "government_agency",
        "news_agency",
        "organization_other",
        "political_party",
        "social_organization",
        "sports_league",
        "sports_team",
    ],
    "person": [
        "politician",
        "musician",
        "actor",
        "artist",
        "athlete",
        "author",
        "businessman",
        "character",
        "coach",
        "director",
        "intellectual",
        "journalist",
        "person_other",
    ],
    "other": [
        "animal",
        "award",
        "medical_thing",
        "website",
        "ordinance",
    ],
    "art": [
        "art_other",
        "film_and_television_works",
        "magazine",
        "music",
        "written_work",
    ],
    "event": [
        "event_other",
        "festival",
        "sports_event",
    ],
    "product": [
        "brand_name_products",
        "game",
        "product_other",
        "software",
    ],
}
FMNERG_FINE_TYPES = ["O"] + [
    fine_type
    for fine_types in FMNERG_COARSE_FINE_TREE.values()
    for fine_type in fine_types
]
FMNERG_FINE_TYPE_TO_ID = {t: i for i, t in enumerate(FMNERG_FINE_TYPES)}
FMNERG_ID_TO_FINE_TYPE = {i: t for i, t in enumerate(FMNERG_FINE_TYPES)}
FMNERG_COARSE_TYPES = ["O"] + list(FMNERG_COARSE_FINE_TREE.keys())
FMNERG_COARSE_TYPE_TO_ID = {t: i for i, t in enumerate(FMNERG_COARSE_TYPES)}
FMNERG_ID_TO_COARSE_TYPE = {i: t for i, t in enumerate(FMNERG_COARSE_TYPES)}
FMNERG_FINE_TO_COARSE = {
    fine_type: coarse_type
    for coarse_type, fine_types in FMNERG_COARSE_FINE_TREE.items()
    for fine_type in fine_types
}


ENTITY_TYPES = COARSE_ENTITY_TYPES
ENTITY_TYPE_TO_ID = COARSE_ENTITY_TYPE_TO_ID
ID_TO_ENTITY_TYPE = COARSE_ID_TO_ENTITY_TYPE


def get_entity_types(task: str) -> list[str]:
    if task == "fmnerg":
        return FMNERG_FINE_TYPES
    return COARSE_ENTITY_TYPES


def get_entity_type_to_id(task: str) -> dict[str, int]:
    if task == "fmnerg":
        return FMNERG_FINE_TYPE_TO_ID
    return COARSE_ENTITY_TYPE_TO_ID


def get_id_to_entity_type(task: str) -> dict[int, str]:
    if task == "fmnerg":
        return FMNERG_ID_TO_FINE_TYPE
    return COARSE_ID_TO_ENTITY_TYPE


def get_num_entity_types(task: str) -> int:
    return len(get_entity_types(task))


def get_coarse_entity_types(task: str) -> list[str]:
    if task == "fmnerg":
        return FMNERG_COARSE_TYPES
    return COARSE_ENTITY_TYPES


def get_coarse_entity_type_to_id(task: str) -> dict[str, int]:
    if task == "fmnerg":
        return FMNERG_COARSE_TYPE_TO_ID
    return COARSE_ENTITY_TYPE_TO_ID


def get_fine_to_coarse_ids(task: str) -> list[int]:
    """Map each task label id to the id of its coarse parent."""
    if task == "fmnerg":
        return [
            (
                0
                if fine_type == "O"
                else FMNERG_COARSE_TYPE_TO_ID[FMNERG_FINE_TO_COARSE[fine_type]]
            )
            for fine_type in FMNERG_FINE_TYPES
        ]
    return list(range(len(COARSE_ENTITY_TYPES)))


def get_coarse_to_fine_transition(task: str) -> list[list[float]]:
    """Build a coarse-to-fine taxonomy prior matrix.

    Rows are coarse labels and columns are task labels. Each non-empty row is
    normalized to sum to 1, with the O row mapped to the O fine label.
    """
    fine_types = get_entity_types(task)
    coarse_types = get_coarse_entity_types(task)
    matrix = [[0.0 for _ in fine_types] for _ in coarse_types]
    if task != "fmnerg":
        for idx in range(min(len(fine_types), len(coarse_types))):
            matrix[idx][idx] = 1.0
        return matrix

    matrix[FMNERG_COARSE_TYPE_TO_ID["O"]][FMNERG_FINE_TYPE_TO_ID["O"]] = 1.0
    for coarse_type, fine_children in FMNERG_COARSE_FINE_TREE.items():
        row = FMNERG_COARSE_TYPE_TO_ID[coarse_type]
        weight = 1.0 / max(len(fine_children), 1)
        for fine_type in fine_children:
            matrix[row][FMNERG_FINE_TYPE_TO_ID[fine_type]] = weight
    return matrix


def get_default_dataset(task: str) -> str:
    if task == "gmner":
        return "Twitter10000_v2.0"
    if task == "fmnerg":
        return "Twitter10000v2"
    return "twitter2017"


def resolve_dataset_split_file(
    data_dir: str,
    dataset: str,
    split: str,
    task: str,
) -> str:
    """Resolve split files across legacy and official GMNER/FMNERG layouts."""
    split_filename = f"{split}.txt"
    base_dir = os.path.join(data_dir, dataset)

    candidates = [os.path.join(base_dir, split_filename)]
    if task == "gmner":
        candidates.insert(0, os.path.join(base_dir, "txt", split_filename))
    elif task == "fmnerg":
        candidates.insert(0, os.path.join(base_dir, "txt_fine", split_filename))
        candidates.append(os.path.join(base_dir, "txt", split_filename))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate

    tried = "\n".join(f"  - {path}" for path in seen)
    raise FileNotFoundError(
        f"Unable to locate split file for task={task!r}, dataset={dataset!r}, split={split!r}.\n"
        f"Tried:\n{tried}"
    )


@dataclass
class ModelConfig:
    # --- Task / visual interface ---
    task: str = "mner"
    visual_backend: str = "raw_image"

    # --- Backbone Encoders ---
    text_model_name: str = "cardiffnlp/twitter-roberta-base"
    image_model_name: str = "openai/clip-vit-base-patch16"
    text_hidden_size: int = 768
    image_hidden_size: int = 768
    vinvl_feature_dim: int = 2048

    # --- Shared Projection ---
    shared_dim: int = 384

    # --- Span Encoder ---
    max_span_length: int = 5
    max_seq_length: int = 128
    max_regions: int = 36

    # --- DreamPRVR-style visual registers (VinVL only) ---
    use_dream_registers: bool = False
    num_visual_registers: int = 4
    use_span_type_registers: bool = False

    # --- H-Index-style grounding decode (VinVL only) ---
    use_region_pointer: bool = False
    grounding_decode_mode: str = "argmax"
    groundable_threshold: float = 0.0

    # --- TIGER-style FMNERG hierarchy ---
    use_hierarchical_fine_logits: bool = False
    coarse_prior_weight: float = 0.3
    use_coarse_fine_transition: bool = False
    transition_prior_weight: float = 0.0

    # --- Proposal-free CLIP patch fallback (VinVL only) ---
    use_clip_patch_fallback: bool = False
    clip_fallback_threshold: float = 0.0

    # --- Lightweight external knowledge / caption injection ---
    knowledge_injection: str = "off"
    knowledge_dropout: float = 0.2
    knowledge_gate_init: float = -2.0

    # --- Grounding calibration v2 ---
    use_type_aware_region_pointer: bool = False

    # --- MQSPN-style recall auxiliary head ---
    use_set_prediction_aux: bool = False
    set_aux_queries: int = 60

    # --- FMNERG dev-time fine-type reranking ---
    fine_rerank_lambda: float = 0.0

    # --- Entity Type ---
    num_types: int = len(COARSE_ENTITY_TYPES)
    num_coarse_types: int = len(COARSE_ENTITY_TYPES)
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

    # --- Classifier / grounding heads ---
    classifier_hidden_dim: int = 256
    width_embed_dim: int = 32
    use_groundable_gate: bool = False


@dataclass
class LossConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.05
    tau: float = 0.07
    label_smoothing: float = 0.02
    r_drop_alpha: float = 0.5

    # --- Reverse imagination loss weights ---
    alpha_rev: float = 1.0
    beta_rev: float = 0.5
    gamma_rev: float = 0.05

    # --- Grounding ---
    grounding_weight: float = 1.0
    groundable_weight: float = 0.0

    # --- DreamPRVR-style auxiliary losses ---
    register_weight: float = 0.1
    qsp_weight: float = 0.05

    # --- FMNERG coarse-fine auxiliary losses ---
    coarse_weight: float = 0.0
    coarse_fine_weight: float = 0.0
    no_region_consistency_weight: float = 0.0
    clip_patch_weight: float = 0.0
    region_hard_negative_weight: float = 0.0
    fine_loss_type: str = "ce"
    fine_focal_gamma: float = 1.5
    fine_class_balance_beta: float = 0.999
    set_aux_weight: float = 0.0


@dataclass
class TrainConfig:
    # --- Data ---
    task: str = "mner"
    dataset: str = "twitter2017"
    data_dir: str = "./data"
    image_dir: str = "./data/images"
    vinvl_dir: Optional[str] = None
    annotation_dir: Optional[str] = None

    # --- Training ---
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    normalize_vinvl: bool = False
    aux_warmup_epochs: int = 0
    set_aux_warmup_epochs: int = 1
    append_caption: bool = False
    caption_files: list[str] = field(default_factory=list)
    caption_max_words: int = 32
    knowledge_files: list[str] = field(default_factory=list)
    knowledge_max_words: int = 32
    knowledge_dropout: float = 0.2

    # --- Optimizer ---
    encoder_lr: float = 2e-5
    new_module_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    # --- Evaluation ---
    save_best: bool = True
    metric_for_best: str = "f1"
    tune_grounding_decode: bool = False
    use_fine_rerank: bool = False
    fine_rerank_lambdas: list[float] = field(default_factory=lambda: [0.0])

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
