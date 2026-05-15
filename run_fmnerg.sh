#!/bin/bash

# Best FMNERG run with VinVL proposals, XML annotations, and external BLIP2 knowledge files.
# Recommended flow:
#   1. python scripts/stage_grounded_assets.py --gmner-src /path/to/Twitter10000_v2.0 \
#        --fmnerg-src /path/to/Twitter10000v2 --vinvl-src /absolute/path/to/Twitter10000_VinVL
#   2. source data/grounded_paths.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATHS_ENV="$ROOT_DIR/data/grounded_paths.sh"
if [ -f "$PATHS_ENV" ]; then
    # shellcheck disable=SC1090
    source "$PATHS_ENV"
fi

DATA_DIR="${DATA_DIR:-${IMAGINE_DATA_DIR:-$ROOT_DIR/data}}"
IMAGE_DIR="${IMAGE_DIR:-${IMAGINE_IMAGE_DIR:-$DATA_DIR/images/Twitter10000}}"
VINVL_DIR="${VINVL_DIR:-${IMAGINE_VINVL_DIR:-$DATA_DIR/vinvl/Twitter10000_shared}}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${IMAGINE_FMNERG_ANNOTATION_DIR:-$DATA_DIR/Twitter10000v2/xml}}"
CAPTION_15="${CAPTION_15:-$ROOT_DIR/thirdparty/PGIM/data/ImageCaption/BLIP2_15.txt}"
CAPTION_17="${CAPTION_17:-$ROOT_DIR/thirdparty/PGIM/data/ImageCaption/BLIP2_17.txt}"

if [ ! -f "$CAPTION_15" ] || [ ! -f "$CAPTION_17" ]; then
    echo "FMNERG knowledge gate requires BLIP2 caption/knowledge files." >&2
    echo "Set CAPTION_15 and CAPTION_17 to absolute paths, for example:" >&2
    echo "  CAPTION_15=/path/to/BLIP2_15.txt CAPTION_17=/path/to/BLIP2_17.txt bash run_fmnerg.sh" >&2
    exit 1
fi

python "$ROOT_DIR/train.py" \
    --task fmnerg \
    --dataset Twitter10000v2 \
    --data_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR" \
    --vinvl_dir "$VINVL_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --visual_backend vinvl \
    --output_dir "$ROOT_DIR/outputs_fmnerg_knowledge_gate" \
    --text_model cardiffnlp/twitter-roberta-base \
    --epochs 20 \
    --batch_size 16 \
    --encoder_lr 2e-5 \
    --new_module_lr 1e-4 \
    --alpha 1.0 --beta 0.5 --gamma 0.05 --tau 0.07 \
    --alpha_rev 0.1 --beta_rev 0.1 --gamma_rev 0.01 \
    --grounding_weight 1.0 \
    --groundable_weight 0.1 \
    --use_region_pointer \
    --grounding_decode_mode soft_groundable \
    --tune_grounding_decode \
    --use_hierarchical_fine_logits \
    --coarse_weight 0.1 \
    --coarse_fine_weight 0.02 \
    --coarse_prior_weight 0.3 \
    --aux_warmup_epochs 1 \
    --knowledge_injection gated_span \
    --knowledge_files "$CAPTION_15" "$CAPTION_17" \
    --knowledge_max_words 32 \
    --knowledge_dropout 0.2 \
    --knowledge_gate_init -2.0
