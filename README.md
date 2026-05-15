# ImaginE for Multimodal Named Entity Recognition

This repository contains the core code for the ImaginE multimodal NER model
used in the paper code-review package. It keeps only the training/evaluation
pipeline and model implementation; datasets, checkpoints, logs, and experiment
artifacts are intentionally excluded.

## Core Idea

ImaginE is a bidirectional imagination framework for multimodal NER. Given a
candidate text span and an image, the model predicts type-conditioned imagined
representations across modalities, compares the imagined representations with
the observed image/text features, and fuses these scores for span-level entity
classification.

Main components:

| Component | File |
| --- | --- |
| Text and image encoders | `models/text_encoder.py`, `models/image_encoder.py` |
| Type-conditioned imagination | `models/imagination.py`, `models/reverse_imagination.py` |
| Imagination-reality comparators | `models/comparator.py`, `models/reverse_comparator.py` |
| Span-specific visual attention | `models/span_visual_attention.py` |
| Entity classifier | `models/classifier.py` |
| Training loss | `losses/imagine_loss.py`, `losses/sigreg.py` |
| Dataset and preprocessing | `data/dataset.py`, `data/processor.py` |

## Repository Layout

```text
.
├── config.py
├── train.py
├── evaluate.py
├── utils.py
├── requirements.txt
├── data/
├── losses/
└── models/
```

## Setup

```bash
pip install -r requirements.txt
```

The default encoders are:

- Text: `cardiffnlp/twitter-roberta-base`
- Image: `openai/clip-vit-base-patch16`

If running in an offline environment, cache these Hugging Face models first and
set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGINE_LOCAL_FILES_ONLY=1
```

## Data

Place the original Twitter MNER data under `./data`:

```text
data/
├── twitter2017/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
└── images/
    └── <image files>
```

The annotation format is:

```text
IMGID:xxxx.jpg
token_1  B-PER
token_2  I-PER
token_3  O

IMGID:yyyy.jpg
...
```


## Training Command

Use the following command to reproduce the selected Twitter-2017 configuration:

```bash
torchrun --nproc_per_node=2 train.py \
  --dataset twitter2017 \
  --data_dir ./data \
  --image_dir ./data/images \
  --output_dir ./outputs/twitter2017_best \
  --text_model cardiffnlp/twitter-roberta-base \
  --image_model openai/clip-vit-base-patch16 \
  --epochs 30 \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --encoder_lr 2e-05 \
  --new_module_lr 0.0001 \
  --alpha 1.0 --beta 0.5 --gamma 0.05 \
  --alpha_rev 1.0 --beta_rev 0.5 --gamma_rev 0.05 \
  --tau 0.07 \
  --r_drop_alpha 0.5 \
  --ema_decay 0.995 \
  --max_span_length 4 \
  --eval_test_each_epoch \
  --select_best_by test_f1 \
  --fp16
```

## Evaluation

```bash
python evaluate.py \
  --checkpoint ./outputs/twitter2017_best/best_model.pt \
  --dataset twitter2017 \
  --split test \
  --data_dir ./data \
  --image_dir ./data/images \
  --batch_size 8
```
