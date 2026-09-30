# Implementation validation: L-Ghost + EMA-32

Validation date: 2026-09-30. Source base: `9f828af` on `feat/yolo12-ema-ghost`.
Environment: Windows, Python 3.12, PyTorch 2.6.0+cpu, ultralytics-thop 2.1.6.
No GPU was available in this implementation environment.

## Results

| Check | Result |
| --- | --- |
| Five focused L-Ghost regression tests | PASS |
| Six existing EMA-Ghost regression tests | PASS |
| Synthetic trainer/interruption/resume/compact reload integration test | PASS |
| Ruff checks for changed Python modules and notebook | PASS |
| Notebook nbformat validation and Python cell compilation | PASS |
| Official YOLO12s checkpoint semantic transfer | PASS |

The integration test creates four small synthetic labeled images per split, trains with the real detection trainer,
interrupts after epoch 2 of a five-epoch run, reloads last.pt, completes the original schedule, verifies both masks reach
stage 3, saves/reloads a compact checkpoint and runs test-split validation. Its accuracy values have no research meaning.
The execution smoke uses random initialization; official pretrained transfer was separately checked against the existing
downloaded official `yolo12s.pt` and semantic regression tests.

Both LG and EMA receive finite, nonzero gradients from a real YOLO detection loss. Gradients on masked weights are exactly
zero. Output shape for two 64x64 images and five classes is `(2, 9, 84)`. Compact FP32 inference matches the masked model
within `rtol=1e-4, atol=1e-4`; the integration test also accounts for FP16 rounding during checkpoint save.

## Official checkpoint audit for the five-class target

| Field | Observed value |
| --- | ---: |
| Source state tensors | 699 |
| Target state tensors | 767 |
| Exactly transferred tensors | 673 |
| Exactly transferred parameter tensors | 340 |
| Exactly transferred parameter elements | 8,515,088 |
| Shape-mismatched tensors | 6 |
| New-module state tensors | 88 |
| Target state tensors not initialized from source | 94 |

State tensor counts include buffers and are not parameter counts. Source/target counts are specific to this revision and
checkpoint. The notebook also writes a fresh actual-trainer audit rather than treating these numbers as proof of a future run.

## Model size and computation estimates

Unfused, five-class models at 640x640, profiled with the same THOP version and attention-counting code:

| Architecture | Parameters | Estimated GFLOPs |
| --- | ---: | ---: |
| Previous YOLO12s + GhostConv + EMA-32 | 8,892,079 | 23.160869 |
| New L-Ghost-inspired + EMA-32, dense masked training form | 8,738,223 | 23.293989 |
| New L-Ghost-inspired + EMA-32, fully-pruned compact form | 8,676,783 | 22.979416 |

The new dense graph reports 314 layers and 8,738,207 gradient-enabled parameter elements when newly constructed.
Model-info's gradient count means `requires_grad=True`; it is not evidence of numerical gradients. Backward tests above
provide that separate evidence. The compact form removes 61,440 LG convolution weights, not 75% of the entire detector.

The dense masked form is slightly more expensive in arithmetic than the old Ghost model despite fewer parameters.
The compact form has slightly fewer estimated operations. Neither figure establishes lower GPU latency or better mAP.
THOP counts the functional masked convolution explicitly; its BN child is counted separately once. Mask application,
input gathering and memory costs are not included in these estimates.

## Limitations and remaining research work

- The Kaggle notebook was structurally validated, not executed end-to-end in Kaggle.
- CUDA, AMP, GPU memory usage and throughput have not been tested here.
- Neither RDD2022-China nor the local research dataset was trained in this implementation run.
- No mAP gain, statistical significance or replication of the paper's reported improvements is claimed.
- The notebook keeps standard best-checkpoint selection; best.pt can predate final pruning. Inspect recorded stages.
- DDP, torch.compile and time-limited/early-stopped pruning schedules are deliberately unsupported.
- Local testing initially encountered unrelated installed TensorBoard/NumPy incompatibility and missing Polars/THOP.
  Isolated test settings and task-local dependencies were used; the user's main environment was not upgraded/downgraded.

Run the notebook with the actual dataset, retain the ZIP and exact commit, and compare under the protocol in
[README_LGHOST_EMA32.md](README_LGHOST_EMA32.md).
