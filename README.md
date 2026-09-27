# Boosting Fine-Grained Visual Counting Using Count-Only Annotations

Code for the ACCV 2026 paper *Boosting Fine-Grained Visual Counting Using
Count-Only Annotations*.

We adapt a pretrained dual-head counter (a density regression head plus a
detection decoder) using **count-only** labels, i.e. global object counts with no
dot or box annotations. Confident detections supply pseudo spatial supervision to
the density head, the density map recalibrates detection confidences, and the two
alternate under count-level constraints with data selection and a Mean Teacher
for stability.

> **Inference head.** Counts are read from the **density integral**
> (`eval_density.py` / `eval_matched.py`), not from thresholded detections. The
> pretrained baseline is strongest through its *detection* head, so any
> comparison must state which head it uses. `eval_matched.py` reports both heads
> from a single forward pass.

## Setup

```bash
conda create -n countex python=3.10.18 && conda activate countex
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Training and inference use `bf16`, so an Ampere or newer NVIDIA GPU is required.

### Environment variables

No credentials are stored in this repository. Export your own:

```bash
export HF_TOKEN=...                 # required, for the model and dataset hubs
export WANDB_API_KEY=...            # optional, omit to disable W&B logging
export HF_HOME=/path/to/hf/cache    # optional
export EXP_ROOT=./experiments       # where checkpoints are written
export DATA_ROOT=./data             # local datasets, if any
```

## Evaluation

`CountEX-KC` and `CountEX-NC-*` are the released stage-1 checkpoints; the stage-2
scripts expect a checkpoint produced by the training step below.

```bash
cd src
# both heads plus a detection-threshold sweep, on val and test
python eval_matched.py --model countex --ckpt_path BBVisual/CountEX-KC \
    --data_split ALL --tag countex_kc

python eval_matched.py --model countex_stage2 \
    --ckpt_path $EXP_ROOT/<run>/best_val_model --data_split ALL --tag ours_kc
```

`--data_split` is `ALL` for the known-category (KC) setting, or one of
`FOO FUN OFF OTR HOU` to hold that supercategory out (NC setting).

Other entry points:

| script | what it measures |
|---|---|
| `eval_density.py` | count from the density integral only |
| `eval.py` | count from thresholded detections only |
| `eval_localization.py` | GAME(0-3) and point P/R/F1 against dot annotations |
| `eval_pairtally_matched.py` | PairTally, both heads, inter/intra breakdown |
| `bench_latency.py` | per-image latency, with and without the training-only prior |

## Training

```bash
cd src
bash scripts/train/kc_stage2.sh          # known-category
bash scripts/train/nc_food_stage2.sh     # hold out Food, and similarly for the rest
```

The recipe is 3 epochs at lr 1e-5 from `BBVisual/CountEX-KC`, with an effective
batch size of 6. The provided scripts use 6 GPUs with `grad_accum 1`; with fewer
GPUs keep the product constant, e.g. 3 GPUs with `grad_accum 2`.

> **Note on the backend.** The DeepSpeed ZeRO-2 configs in `ddp_cfgs/` do not
> support gradient accumulation across processes in this environment
> (`no_sync ... incompatible with gradient partitioning logic of ZeRO stage 2`).
> If you need `grad_accum > 1`, use the plain-DDP configs (`*_ddp.yaml`) instead,
> and pass `--save_safetensors False`, since the model ties `class_embed`
> weights between the decoder and the top level.
