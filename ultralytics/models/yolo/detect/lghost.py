"""Opt-in trainer for the L-Ghost-inspired detection variant; standard trainers remain unchanged."""

import json
import math

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.lghost import CompactLearnedGroupConv, LearnedGroupConv
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def pruning_stage(completed_epochs: int, epochs: int, groups: int, ratio: float):
    """Return the stage after ceil(RA * M * stage / (G-1)) completed epochs."""
    if epochs < 1 or groups < 2 or not 0 < ratio < 1:
        raise ValueError("Require epochs >= 1, groups >= 2 and 0 < pruning ratio < 1")
    return sum(completed_epochs >= math.ceil(ratio * epochs * stage / (groups - 1)) for stage in range(1, groups))


def update_pruning(trainer):
    """Advance masks at epoch boundaries and copy discrete state into the weight-EMA evaluation model."""
    model = unwrap_model(trainer.model)
    ratio = model.yaml["lghost_pruning"]["ratio"]
    shadow = dict(trainer.ema.ema.named_modules()) if trainer.ema else {}
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, LearnedGroupConv):
            continue
        stage = pruning_stage(trainer.epoch, trainer.epochs, module.groups, ratio)
        previous = int(module.stage)
        module.prune_to_stage(stage)
        if name in shadow:
            # Ultralytics ModelEMA averages floating buffers only. Never average a pruning mask.
            shadow[name].mask.copy_(module.mask)
            shadow[name].stage.copy_(module.stage)
        if stage != previous:
            records.append({"module": name, "stage": stage, "retained_fraction": float(module.mask.float().mean())})
    if records:
        LOGGER.info(f"L-Ghost pruning before epoch {trainer.epoch + 1}: {records}")
        with (trainer.save_dir / "lghost_pruning.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"completed_epochs": trainer.epoch, "blocks": records}) + "\n")


def audit_lghost_setup(trainer):
    """Fail early on unsupported execution or invalid schedules; record actual pretrained coverage."""
    if trainer.world_size > 1:
        raise ValueError("L-Ghost trainer currently supports one GPU or CPU only; use device=0")
    if trainer.args.compile or trainer.args.time or trainer.args.patience:
        raise ValueError("Use compile=False, time=None and patience=0 for the fixed L-Ghost pruning schedule")
    model = unwrap_model(trainer.model)
    ratio = model.yaml.get("lghost_pruning", {}).get("ratio")
    blocks = [m for m in model.modules() if isinstance(m, LearnedGroupConv)]
    if not blocks or not isinstance(ratio, (float, int)):
        raise ValueError("L-Ghost YAML must contain LearnedGroupConv blocks and lghost_pruning.ratio")
    for module in blocks:
        pruning_stage(0, trainer.epochs, module.groups, ratio)
        if math.ceil(ratio * trainer.epochs) >= trainer.epochs:
            raise ValueError("Epoch budget must leave at least one training epoch after final pruning")
    report = getattr(model, "pretrained_transfer_report", {})
    (trainer.save_dir / "pretrained_transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(f"L-Ghost pretrained transfer: {report}")


class LGhostDetectionTrainer(DetectionTrainer):
    """Standard detection optimizer/loss plus scheduled LG pruning and exact weight-EMA mask synchronization."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Attach only the callbacks required by this experimental architecture."""
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_pretrain_routine_end", audit_lghost_setup)
        self.add_callback("on_train_epoch_start", update_pruning)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Reject compact inference checkpoints as training initialization."""
        if weights is not None and any(isinstance(m, CompactLearnedGroupConv) for m in weights.modules()):
            raise ValueError("Compact checkpoints are inference-only; train/resume from masked best.pt or last.pt")
        return super().get_model(cfg, weights, verbose)
