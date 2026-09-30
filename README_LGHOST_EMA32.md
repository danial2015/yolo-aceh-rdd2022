# YOLO12s + L-Ghost-inspired blocks + EMA-32

Experimental road-damage detector for **separate** RDD2022-China and local-dataset studies.
This branch preserves the previous architectures and notebooks. It does not assert improved mAP.

| Item | Value |
| --- | --- |
| Repository | [danial2015/yolo-aceh-rdd2022](https://github.com/danial2015/yolo-aceh-rdd2022) |
| Branch | `feat/yolo12s-lghost-ema32` |
| Source branch / base | `feat/yolo12-ema-ghost` / `9f828af` |
| Model YAML | [yolo12s-lghost-ema32.yaml](ultralytics/cfg/models/12/yolo12s-lghost-ema32.yaml) |
| Kaggle notebook | [kaggle_yolo12s_lghost_ema32_rdd2022.ipynb](examples/kaggle_yolo12s_lghost_ema32_rdd2022.ipynb) |
| Modules | [lghost.py](ultralytics/nn/modules/lghost.py) |
| Required training integration | [LGhostDetectionTrainer](ultralytics/models/yolo/detect/lghost.py) |

## Research scope and attribution

Reference: Jing Chi, Shaohua Guo, Haopeng Zhang, and Yu Shan, **L-GhostNet: Extract Better Quality Features**,
IEEE Access 11 (2023), 2361-2374. DOI: [10.1109/ACCESS.2023.3234108](https://doi.org/10.1109/ACCESS.2023.3234108).
The implementation is an independent adaptation of sections III.A/III.C and Figure 3, not author-released code.

The paper replaces the first Ghost module in a Ghost bottleneck with learned group convolution (LG), and introduces
p-CA attention. Its experiments are classification experiments; their accuracy/efficiency gains cannot be claimed for YOLO.
Here we adopt LG column pruning and the LG/DW/Ghost residual structure, but **omit p-CA** and retain the existing external
EMA-32 attention. Do not label this model a full L-GhostNet reproduction.

Explicit adaptation choices:

- Only two bottom-up neck convolutions change. Backbone, native A2C2f/C3k2 and detection scales stay unchanged.
- Hidden LG width equals the output width; this is not the full L-GhostNet Table 1 expansion schedule.
- LG uses BN/ReLU; the existing Ultralytics Ghost projection, depthwise and residual primitives are reused.
- No extra internal attention is stacked on EMA. EMA stays at the original positions, not the p-CA position in Figure 3.
- `G=4`, `RA=0.6` are initial method settings inspired by the paper, not tuned optima for road damage.
- Groups in LG (`G=4`) are unrelated to the 32 groups of Efficient Multi-scale Attention.

## Before and after

Layer numbers are zero-based in the constructed small model.

| Layer | Previous EMA-Ghost variant | New variant | Actual small-model channels |
| --- | --- | --- | --- |
| 0-14 | YOLO12s backbone and top-down fusion | Unchanged | Unchanged |
| 15 | EMA-32 on P3 | Unchanged | 128 |
| 16 | GhostConv, stride 2 | LGhostBottleneck, stride 2, G=4 | 128 -> 128 |
| 17-19 | P4 fusion and EMA-32 | Unchanged | P4: 256 |
| 20 | GhostConv, stride 2 | LGhostBottleneck, stride 2, G=4 | 256 -> 256 |
| 21-23 | P5 fusion and Detect | Unchanged | Detect from [15,19,22] |

Each replacement computes `LG -> stride-2 DWConv -> Ghost projection`, added to a depthwise/pointwise shortcut.
This is a larger block than a single GhostConv; it is not automatically cheaper than the old detector.

## Learned pruning and checkpoint correctness

LG starts with dense `O x I x 1 x 1` weights. Output channels are divided into G equal groups.
Within each group, input columns are ranked by the sum of absolute weights across its output channels.
The lowest-scoring **still-active** I/G columns are removed per stage. Selection is independent across groups.

Pruning stage k starts after `ceil(RA * epochs * k / (G-1))` completed epochs. For 160 epochs and G=4:

| Training epochs (one-based) | Stage | Retained LG input columns per group |
| --- | --- | --- |
| 1-32 | 0 | 100% |
| 33-64 | 1 | 75% |
| 65-96 | 2 | 50% |
| 97-160 | 3 | 25% |

Masks are boolean buffers and stages are integer buffers saved in checkpoints. Masks remain in forward: momentum or weight
decay cannot reactivate a removed connection. LG weights are still dense allocations while training.
The opt-in trainer explicitly copies masks/stages to Ultralytics **ModelEMA** (exponential moving average of weights).
This ModelEMA is separate from **EMA-32 attention**. Their names must not be conflated.

Use `LGhostDetectionTrainer` for every train/resume call. The standard trainer alone does **not** run the pruning schedule.
Supported execution is one GPU or CPU, `compile=False`, `time=None`, `patience=0` and a fixed epoch budget that leaves
at least one epoch after final pruning. DDP/compiled/time-limited training are rejected rather than silently mis-scheduled.
Resume with the original epoch budget, YAML ratio, group count and repository revision.

## Pretrained YOLO12s: what transfers

The notebook downloads the official `yolo12s.pt`, then passes `pretrained="yolo12s.pt"` to the actual training call.
The existing semantic source-to-target map is reused: shifted but unchanged neck/detection layers receive the right weights.
Original baseline Conv layers 15/18 are excluded because their replacements have different semantics.

The backbone and compatible neck/head tensors transfer. LG, replacement bottlenecks, EMA attention and incompatible
five-class logits are newly initialized. No checkpoint can supply pretrained LG/EMA weights it never contained.
The preflight prints the architecture/transfer report; the trainer separately audits and saves the **actual** transfer.
Weights are transferred, not gradients: new gradients are computed from the detection loss during training.

## Kaggle workflow

1. Open the new notebook, enable Internet and one GPU, and start a clean session.
2. Set `DATA_ROOT` to your attached dataset. Default follows the previous notebook's
   `/kaggle/input/datasets/danialalfayyadh/ch-rdd-2022/datasets-china-split-fix`.
   The earlier `.../ch-rdd2022/datasets-china-split` path is included as an alternative comment.
3. Confirm train/val/test images and YOLO labels exist. The five classes are D00, D10, D20, D40 and Repair.
   Update names/path for local data as necessary; never mix the two datasets into a cross-dataset run.
4. Run clone/install, config, pretrained inspection, training, validation, test and ZIP cells in order.
5. Download/save Kaggle outputs before ending the session. `/kaggle/working` is not persistent backup by itself.

The notebook uses the new branch, records the resolved commit and Python-package versions, verifies the imported package
is from that clone, and refuses to overwrite a dirty/different checkout. For exact reproduction, use the recorded commit.

Common hyperparameters follow the source notebook: 160 epochs, 640 pixels, batch 16, nbs 64, SGD, lr0 0.01,
momentum 0.937, weight decay 0.0005, seed 42, workers 2, cos_lr False, patience 0. Other defaults come from the recorded
repo revision and are stored in `train/args.yaml`. This is not a claim that the complete classification-paper protocol is used.
Validation still runs every epoch. Only screen output is summarized at epoch 1, every 20 epochs and the final epoch;
all metrics remain in `results.csv`, and detailed messages go to `training.log`.

The ZIP includes best/last checkpoints, per-epoch metrics, pruning events, transfer report, full log, val/test outputs,
model print, YAML, commit/environment metadata and a notebook copy. The ZIP cell may also be run after an interruption.

## Compact inference and evaluation fairness

`compact_lghost(model)` deep-copies a fully-pruned model and replaces masked LG with input gather + regular grouped 1x1
convolution. Different groups may retain overlapping input channels; gathering preserves this learned connectivity.
It physically reduces LG convolution weights by G. BN, Ghost projections and shortcuts are unchanged.
The conversion refuses incomplete pruning and leaves the original model/checkpoints untouched.

`best.pt` keeps the standard Ultralytics validation-fitness selection across **all** epochs. It may come from before the final
pruning stage. The notebook reports its stages and skips compact conversion in that case; it never silently prunes the best
checkpoint or swaps it for last.pt. `last.pt` is fully pruned after an uninterrupted complete run, but is a different checkpoint.
Any comparison using last.pt or post-pruning-only selection must be explicitly identified and applied consistently to baselines.

Compact checkpoints are inference-only in this integration; use un-compacted checkpoints for training/resume. The notebook
checks FP32 output equivalence before saving. Saved Ultralytics checkpoints may round to FP16. Validation metrics in the main
report belong to the original best.pt, not an unmeasured compact variant.

THOP model-info includes the dense masked LG convolution cost, even after masking; it does not pretend zeros reduce dense
MACs. Compact grouped convolutions are counted normally. These are arithmetic estimates, excluding mask/gather overhead:
measure actual latency/VRAM on the same hardware. Reduced LG weights do not imply a guaranteed whole-model speedup.

## Verification commands

From this checkout, with its dependencies installed:

```bash
python -m unittest tests.test_yolo12_lghost tests.test_yolo12_ema_ghost -v
python -m unittest tests.test_yolo12_lghost_integration -v
```

Focused tests cover geometry, real detection-loss backward, nonzero LG/EMA gradients, monotonic masks, zero masked gradients,
optimizer momentum, schedule boundaries, semantic pretrained mapping, weight-EMA synchronization, serialization, compact
equivalence and synthetic train/interruption/resume/validation. The synthetic test is not a measurement of research mAP.
See [VALIDATION_LGHOST.md](VALIDATION_LGHOST.md) for results and limitations from this implementation run.

## Suggested research comparison

Keep YOLO12s original and the previous GhostConv+EMA-32 model as controls. Use identical splits, training budgets, generic
hyperparameters, validation selection and preferably multiple seeds for each dataset separately. Report mAP50, mAP50-95,
per-class results, parameters, computation, latency and peak VRAM. Disclose the added pruning schedule and pretrained
coverage differences. Do not select a model using test-set performance or report unmeasured improvements.
