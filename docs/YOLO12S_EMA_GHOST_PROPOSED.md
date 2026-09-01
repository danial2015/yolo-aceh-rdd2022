# YOLO12s + EMA(P3/P4) + Selective GhostConv

## Scope

This branch implements one proposed architecture for experimental evaluation:

```text
YOLO12s + direct EMA attention on final fused P3/P4 + selective GhostConv on bottom-up downsampling
```

It does not change the YOLO12s backbone, early convolution layers, native `A2C2f` implementation, dataset code,
training protocol, loss, optimizer, or the original `yolo12.yaml`. No training or mAP claim is included in this
implementation report.

## Architecture implemented

The original YOLO12s backbone remains unchanged. The final P3 and P4 head features are refined with direct Efficient
Multi-scale Attention (EMA). Only the two bottom-up stride-2 convolutions are replaced by the repository's existing
`GhostConv`.

| Role | Baseline YOLO12s | Proposed layer | Output at 640 |
| --- | --- | ---: | --- |
| Final fused P3 | `A2C2f` at 14 | 14 | `128 × 80 × 80` |
| EMA-P3 | — | 15: `EMA(128, 32)` | `128 × 80 × 80` |
| P3 → P4 downsampling | `Conv` at 15 | 16: `GhostConv(128, 128, 3, 2)` | `128 × 40 × 40` |
| Bottom-up P4 concat | `Concat` at 16 | 17 | `384 × 40 × 40` |
| Final fused P4 | `A2C2f` at 17 | 18 | `256 × 40 × 40` |
| EMA-P4 | — | 19: `EMA(256, 32)` | `256 × 40 × 40` |
| P4 → P5 downsampling | `Conv` at 18 | 20: `GhostConv(256, 256, 3, 2)` | `256 × 20 × 20` |
| Bottom-up P5 concat | `Concat` at 19 | 21 | `768 × 20 × 20` |
| Final P5 | `C3k2` at 20 | 22 | `512 × 20 × 20` |
| Detection inputs | `[14, 17, 20]` | 23: `[15, 19, 22]` | P3, P4, P5 |

The actual head channels for scale `s` are P3=`128` and P4=`256`. Both satisfy the selected EMA grouping factor:
`128 % 32 = 0` and `256 % 32 = 0`.

There is deliberately no EMA module at P5, and no GhostConv in the stem or early backbone.

## Source integration

| File | Change |
| --- | --- |
| `ultralytics/nn/modules/conv.py` | Adds public `EMA` using direct grouped Efficient Multi-scale Attention. Its affinity softmax and matrix products use FP32 under AMP, while the returned feature keeps the input dtype. |
| `ultralytics/nn/modules/__init__.py` | Re-exports `EMA` from the stable module namespace. |
| `ultralytics/nn/tasks.py` | Adds the shape-preserving EMA parser rule and a validated semantic state-dict remap in `BaseModel.load()`. |
| `ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml` | Adds the single proposed graph and its declarative baseline-to-proposed layer map. |
| `ultralytics/engine/model.py` | Exposes the load report as `YOLO(...).pretrained_transfer_report`. |
| `tests/test_yolo12_ema_ghost.py` | Adds focused structural and semantic-initialization regression tests. |

The proposed model can be instantiated with:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml")
model.load("yolo12s.pt")
print(model.pretrained_transfer_report)
model.info()
```

The same mapping is used by the normal training path when the checkpoint is explicitly selected, for example
`model.train(data="data.yaml", pretrained="yolo12s.pt")`. Bare `pretrained=True` does not select a source checkpoint
for a model constructed only from YAML.

## Structural validation

Validation was run from this branch using the repository package from the active worktree on CPU.

| Check | Result | Evidence |
| --- | --- | --- |
| Python syntax | PASS | `python -m compileall` completed for every changed Python file. |
| EMA standalone | PASS | `EMA(128, 32)` and `EMA(256, 32)` preserve shape and have finite backward gradients. Invalid channel/group combinations raise `ValueError`. |
| AMP-oriented CPU smoke check | PASS | bfloat16-autocast forward/backward produced finite loss and input gradients at both P3 and P4 channel counts. |
| YAML parse and model build | PASS | `DetectionModel` resolves `EMA`, `GhostConv`, and the new parser semantics. |
| Focused test suite | PASS | Six `unittest` tests pass, including sentinel-based semantic-transfer checks for `nc=80`, `nc=5`, and an already-proposed checkpoint. |
| 640 dummy forward | PASS | A 5-class build returns prediction shape `(1, 9, 8400)` and the exact feature shapes in the architecture table. |
| Detect routing | PASS | Detection head sources are `[15, 19, 22]`: EMA-refined P3, EMA-refined P4, and unmodified final P5. |
| Semantic pretrained transfer | PASS | The official `yolo12s.pt` maps P4, P5, and Detect by architecture semantics rather than their old numeric indices. |
| Checkpoint serialization | PASS | A temporary proposed checkpoint reloads through both `load_checkpoint()` and safe loading with class path `ultralytics.nn.modules.conv.EMA`. |

The local Python environment is CPU-only and lacks installed `torchvision` distribution metadata. A temporary,
process-local metadata workaround was used only to execute the repository validation; it is not committed to source or
needed by a normally installed Ultralytics environment.

## Semantic pretrained initialization

The original key-and-shape loader retained only `452/715` target state entries. Of those, 450 were valid unchanged
modules, while two `model.20.*.bn.num_batches_tracked` counters were an accidental collision between the baseline P5
`C3k2` and the proposed P4→P5 GhostConv. It also missed the unchanged P4 fusion, P5 fusion, and Detect weights because
their numeric indices shifted.

The proposed YAML now declares this source-to-target semantic graph map:

| Baseline YOLO12s | Proposed YOLO12s EMA-Ghost | Reason |
| ---: | ---: | --- |
| 0–14 | 0–14 | Unchanged backbone and top-down/P3 path. |
| 16 | 17 | Bottom-up P4 `Concat`; no parameters, retained as graph validation. |
| 17 | 18 | Final fused P4 `A2C2f`; all compatible tensors are retained. |
| 19 | 21 | Bottom-up P5 `Concat`; no parameters, retained as graph validation. |
| 20 | 22 | Final P5 `C3k2`; all compatible tensors are retained. |
| 21 | 23 | Detect; regression and compatible classification tensors are retained. |

Baseline layers 15 and 18 are intentionally absent: their stride-2 `Conv` modules are replaced by GhostConv and have
no exact one-to-one tensor mapping. EMA layers 15 and 19 are also new.

The semantic policy activates only when the source has the expected 22-layer YOLO12 detection graph, Detect inputs
`[14, 17, 20]`, and matching module types for every declared pair. If the source is already a proposed checkpoint or
does not match this fingerprint, loading falls back to the repository's standard key-and-shape behavior. The remapped
state dictionary is filtered—not merged with the raw source keys—so the two incorrect GhostConv BatchNorm counter
collisions cannot recur.

### Actual official-checkpoint results

The following measurements use the released `yolo12s.pt` checkpoint from the official assets release. It has 699 state
entries; eight legacy attention positional-convolution bias entries at baseline layers 6/8 have no counterpart in the
current repository source and are excluded independently of this proposed architecture.

| Target configuration | Exact tensors from official source | Target state retained | Exact target parameter tensors retained | Exact parameter elements retained |
| --- | ---: | ---: | ---: | ---: |
| Architecture-equivalent `nc=80` | 679 / 699 (97.14%) | 679 / 715 (94.97%) | 346 / 370 | 8,546,048 / 8,921,104 |
| Five-class fine-tuning `nc=5` | 673 / 699 (96.28%) | 673 / 715 (94.13%) | 340 / 370 | 8,515,088 / 8,892,079 |

For `nc=80`, the 36 target entries left fresh are exactly the two EMA modules (6 entries each) and two GhostConv
modules (12 entries each). For `nc=5`, six final Detect class-logit tensors are additionally fresh because they change
from 80 COCO classes to five dataset classes; the remaining 115 Detect tensors still transfer exactly.

A baseline reconstructed by this checkout has 691 state entries rather than the released checkpoint's 699 legacy
entries. The semantic map transfers 679 / 691 (98.26%) from that current baseline layout: the only twelve unmapped
source entries are the two replaced baseline stride-2 Conv modules. Focused tests fill every source tensor with a
distinct sentinel and prove both that every approved mapping is byte-identical after loading and that every unapproved
target tensor remains at its fresh initialization.

## Structural summary

Both rows use `nc=80` and input `640 × 640`. The baseline was constructed from `yolo12.yaml` with `scale='s'`; the
proposed YAML explicitly sets `scale: s`.

| Model | Layers | Parameters | Gradients | CPU profiler estimate |
| --- | ---: | ---: | ---: | ---: |
| YOLO12s baseline | 272 | 9,284,096 | 9,284,080 | 23.349 GFLOPs |
| YOLO12s EMA-Ghost proposed | 286 | 8,921,104 | 8,921,088 | 23.012 GFLOPs |

`model.info()` completed for both models. It reported the layer and parameter counts above; it did not emit GFLOPs in
this local environment because the optional `ultralytics-thop` package is not installed. The final column is a
CPU `torch.profiler` estimate, included only as a structural reference rather than a benchmark.

## Frozen research snapshot

The architecture and pretrained-initialization policy are frozen at the annotated Git tag
`research/yolo12s-ema-ghost-architecture-v1`, which resolves to commit
`d199dd3bac9dd70875370ed3ba16a8a22da1a21b` on branch `feat/yolo12-ema-ghost` (2026-09-01).

This immutable snapshot contains the direct EMA(P3/P4), selective GhostConv, declarative semantic map, and
regression tests described above. Subsequent commits on the branch are limited to reproducible training artifacts
and documentation; they must not modify the architecture or semantic transfer policy. The Kaggle runner clones this
tag explicitly, rather than the moving branch tip, so each proposed-model training run records the same source.

## Files and Git

The implementation is on branch `feat/yolo12-ema-ghost`, which tracks
`origin/feat/yolo12-ema-ghost` in the configured repository.

## Remaining issues

None for source integration and structural validation. Full training, validation metrics, and comparison with the
baseline remain separate experimental work.
