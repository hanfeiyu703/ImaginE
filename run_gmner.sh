#!/bin/bash

# Best GMNER run with VinVL proposals and XML grounding annotations.
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
ANNOTATION_DIR="${ANNOTATION_DIR:-${IMAGINE_GMNER_ANNOTATION_DIR:-$DATA_DIR/Twitter10000_v2.0/xml}}"

python "$ROOT_DIR/train.py" \
    --task gmner \
    --dataset Twitter10000_v2.0 \
    --data_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR" \
    --vinvl_dir "$VINVL_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --visual_backend vinvl \
    --output_dir "$ROOT_DIR/outputs_gmner_type_aware_grounding" \
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
    --use_type_aware_region_pointer \
    --region_hard_negative_weight 0.05
