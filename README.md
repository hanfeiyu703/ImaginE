# TCHV

This repository contains a span-based TCHV implementation for multimodal named entity recognition and grounding.

Supported tasks:

- **MNER**: multimodal named entity recognition.
- **GMNER**: grounded multimodal named entity recognition.
- **FMNERG**: fine-grained multimodal named entity recognition and grounding.

## Training

### MNER

```bash
python train.py \
    --dataset twitter2017 \
    --data_dir ./data \
    --image_dir ./data/images \
    --output_dir ./outputs
```

### GMNER

```bash
bash run_gmner.sh
```

Set `DATA_DIR`, `IMAGE_DIR`, `VINVL_DIR`, or `ANNOTATION_DIR` if your data is not under `./data`.

### FMNERG

```bash
bash run_fmnerg.sh
```

Set `CAPTION_15` and `CAPTION_17` if the BLIP2 knowledge files are not in the default location.
