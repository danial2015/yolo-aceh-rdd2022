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
| `ultralytics/nn/tasks.py` | Adds the smallest shape-preserving parser rule: `EMA(channels_in, factor)` with `channels_out = channels_in`. It is not placed in `base_modules` or `repeat_modules`. |
| `ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml` | Adds the single proposed YOLO12s graph. |
| `tests/test_yolo12_ema_ghost.py` | Adds focused, dependency-light structural tests. |

The proposed model can be instantiated with:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml")
model.info()
```

## Structural validation

Validation was run from this branch using the repository package from the active worktree on CPU.

| Check | Result | Evidence |
| --- | --- | --- |
| Python syntax | PASS | `python -m compileall` completed for every changed Python file. |
| EMA standalone | PASS | `EMA(128, 32)` and `EMA(256, 32)` preserve shape and have finite backward gradients. Invalid channel/group combinations raise `ValueError`. |
| AMP-oriented CPU smoke check | PASS | bfloat16-autocast forward/backward produced finite loss and input gradients at both P3 and P4 channel counts. |
| YAML parse and model build | PASS | `DetectionModel` resolves `EMA`, `GhostConv`, and the new parser semantics. |
| Focused test suite | PASS | Three `unittest` tests pass: EMA behavior, group validation, and graph/64-pixel forward geometry. |
| 640 dummy forward | PASS | A 5-class build returns prediction shape `(1, 9, 8400)` and the exact feature shapes in the architecture table. |
| Detect routing | PASS | Detection head sources are `[15, 19, 22]`: EMA-refined P3, EMA-refined P4, and unmodified final P5. |
| Checkpoint serialization | PASS | A temporary proposed checkpoint reloads through both `load_checkpoint()` and safe loading with class path `ultralytics.nn.modules.conv.EMA`. |

The local Python environment is CPU-only and lacks installed `torchvision` distribution metadata. A temporary,
process-local metadata workaround was used only to execute the repository validation; it is not committed to source or
needed by a normally installed Ultralytics environment.

## Pretrained checkpoint behavior

The official `yolo12s.pt` checkpoint was tested with the repository's normal `BaseModel.load()` mechanism. The result
was `452/715` exact key-and-shape state entries transferred.

This is intentionally **partial** transfer, not full pretrained continuity:

- 450 transferred entries belong to unchanged parameterized layers through the retained backbone and the unchanged
  indexed P3-path modules.
- The two newly inserted EMA modules and two GhostConv replacements are initialized as new modules; their learnable
  tensors do not have a one-to-one source tensor in `yolo12s.pt`.
- P4/P5 fusion and detection layers are renumbered after the EMA insertions, so the normal key-and-shape loader does
  not forcibly map their later baseline weights into different modules.
- The remaining two matching entries are BatchNorm `num_batches_tracked` counters whose names happen to overlap at
  layer 20; they are bookkeeping counters, not learnable GhostConv weights.

This behavior uses the official loader without custom remapping or forced tensor reshaping. It is therefore safe to
start a fine-tuning run with compatible pretrained tensors while keeping the proposed modules genuinely new.

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

## Files and Git

The implementation is on branch `feat/yolo12-ema-ghost` and is intended to be pushed to the configured `origin`
repository after final review.

## Remaining issues

None for source integration and structural validation. Full training, validation metrics, and comparison with the
baseline remain separate experimental work.
