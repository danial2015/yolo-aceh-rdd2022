# YOLO12s Road-Damage Detection

## Final Research Registry

**Branch:** `research/yolo12-rdd2022-final`
**Status:** Experiment execution in progress; numerical results are intentionally not reported in this document yet.

This branch is the single documentation entry point for the final proposed-model study. It does not replace, delete, or merge the historical implementation branches. Each architecture remains linked to its exact source revision so that a completed experiment can be reproduced without ambiguity.

## Research Objective

Evaluate whether lightweight architectural modifications improve the road-damage detection performance of YOLO12-Small under a controlled training protocol.

The study compares one unmodified baseline with three proposed variants:

1. YOLO12s original.
2. YOLO12s + Efficient Multi-scale Attention with factor 32 (EMA-32).
3. YOLO12s + GhostConv.
4. YOLO12s + GhostConv + EMA.

This registry defines the final study scope only. Other experimental branches, including ECA, SPD-Conv, BL-YOLOv8, and intermediate fixes, are retained as historical work but are not part of the final proposed-model comparison unless this document is explicitly revised.

## Architecture Registry

| ID | Architecture | Source revision | Model configuration | Design role |
|---|---|---|---|---|
| B0 | YOLO12s original | Official [Ultralytics](https://github.com/ultralytics/ultralytics) checkpoint `yolo12s.pt` | Official YOLO12s checkpoint | Baseline |
| P1 | YOLO12s + EMA-32 | Branch [`yolo12-ema32`](https://github.com/danial2015/yolo-aceh-rdd2022/tree/yolo12-ema32), commit [`419ee77`](https://github.com/danial2015/yolo-aceh-rdd2022/commit/419ee77d4c9dc716f312d5569972252897a55cf6) | `ultralytics/cfg/models/12/yolo12-ema32.yaml` | Attention ablation |
| P2 | YOLO12s + GhostConv | Branch [`yolo12-ghost-conv`](https://github.com/danial2015/yolo-aceh-rdd2022/tree/yolo12-ghost-conv), commit [`57ea1b4`](https://github.com/danial2015/yolo-aceh-rdd2022/commit/57ea1b42c9020c28db85aed078dd96d9c7ef6dd0) | `ultralytics/cfg/models/12/yolo12-ghost.yaml` | Lightweight convolution ablation |
| P3 | YOLO12s + GhostConv + EMA | Immutable tag [`research/yolo12s-ema-ghost-architecture-v1`](https://github.com/danial2015/yolo-aceh-rdd2022/tree/research/yolo12s-ema-ghost-architecture-v1), commit [`d199dd3`](https://github.com/danial2015/yolo-aceh-rdd2022/commit/d199dd3bac9dd70875370ed3ba16a8a22da1a21b) | `ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml` | Combined proposed model |

The P3 tag is the authoritative source for the GhostConv + EMA architecture. Its originating development branch is [`feat/yolo12-ema-ghost`](https://github.com/danial2015/yolo-aceh-rdd2022/tree/feat/yolo12-ema-ghost), but the immutable tag and commit above must be cited for reproducibility.

## Dataset Protocol

Two datasets are evaluated as independent experiments.

| Dataset | Classes | Intended use | Rule |
|---|---:|---|---|
| RDD2022 - China | `D00`, `D10`, `D20`, `D40`, `Repair` | Training, validation, and test within the China dataset split | Do not combine with the local dataset. |
| Local Aceh dataset | `D00`, `D10`, `D20`, `D40` | Training, validation, and test within the local dataset split | Local source path: `E:\dataset_training_ai\datasets_here`. Do not combine with RDD2022-China. |

The study is **not** a cross-dataset experiment. In particular:

- No model is trained on China and tested on Aceh.
- No model is trained on Aceh and tested on China.
- Images or annotations from the two datasets are never merged.
- mAP values are compared fairly only between models evaluated on the same dataset, class set, split, and training protocol.

Because the class sets differ, China and Aceh results must be presented in separate tables. A single pooled mAP across the two datasets is not a valid study metric.

## Controlled Training Protocol

Unless an experiment record states otherwise, all four architectures should use the following common settings within each dataset:

| Setting | Value |
|---|---:|
| Initialization | `yolo12s.pt` pretrained weights |
| Epochs | 160 |
| Input resolution | 640 |
| Physical batch size | 16 |
| Nominal batch size (`nbs`) | 64 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Seed | 42 |
| Early-stopping patience | 0 |

For proposed models, compatible YOLO12s tensors are transferred before fine-tuning. Newly introduced GhostConv or EMA tensors and the task-specific detection head remain trainable and are initialized for the new task. Every run must archive its pretrained-transfer report; it is evidence of the initialization policy, not a performance metric.

## Results Register

Populate this register only after evaluating `best.pt` on the corresponding validation and test split. Do not substitute training-log values for the final evaluation output.

### RDD2022 - China

| ID | Status | Best epoch | Validation mAP@50 | Validation mAP@50:95 | Test mAP@50 | Test mAP@50:95 |
|---|---|---:|---:|---:|---:|---:|
| B0 | Pending | N/A | N/A | N/A | N/A | N/A |
| P1 | Pending | N/A | N/A | N/A | N/A | N/A |
| P2 | Pending | N/A | N/A | N/A | N/A | N/A |
| P3 | Pending | N/A | N/A | N/A | N/A | N/A |

### Local Aceh Dataset

| ID | Status | Best epoch | Validation mAP@50 | Validation mAP@50:95 | Test mAP@50 | Test mAP@50:95 |
|---|---|---:|---:|---:|---:|---:|
| B0 | Pending | N/A | N/A | N/A | N/A | N/A |
| P1 | Pending | N/A | N/A | N/A | N/A | N/A |
| P2 | Pending | N/A | N/A | N/A | N/A | N/A |
| P3 | Pending | N/A | N/A | N/A | N/A | N/A |

## Required Run Artifacts

Each experiment must produce one ZIP archive containing:

- `best.pt` and `last.pt` checkpoints;
- the model YAML used for the run;
- the generated dataset YAML and dataset audit report;
- repository URL, branch or tag, and full commit SHA;
- pretrained-transfer report;
- validation and test metric JSON files;
- `results.csv`, figures, and confusion matrix generated by Ultralytics.

Record the ZIP filename, execution date, hardware, and Kaggle notebook version alongside each results-table entry. This keeps the final paper traceable without committing large binary artifacts into Git.

## Branch Governance

- `research/yolo12-rdd2022-final` is the final study registry and documentation branch.
- The implementation branches and immutable P3 tag listed above are preserved unchanged.
- Results are added only after a complete run finishes and its artifacts have been checked.
- Any change to a model YAML, dataset split, class mapping, pretrained initialization, or hyperparameter requires a new experiment identifier rather than overwriting an existing result.
- When the final results are accepted, create a release tag from this branch, for example `final-yolo12-rdd2022-v1`.

## Citation and Reporting Language

Use the following wording in a report before results are available:

> We evaluate the original YOLO12-Small baseline and three controlled variants: EMA-32, GhostConv, and GhostConv combined with EMA. Experiments on RDD2022-China and the local Aceh dataset are conducted independently; no cross-dataset training, testing, or data merging is performed.

This statement describes the protocol only. Performance-improvement claims must be made only after the result tables above are filled from completed, reproducible runs.
