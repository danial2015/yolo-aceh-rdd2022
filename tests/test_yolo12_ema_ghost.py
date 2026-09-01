# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused structural tests for the YOLO12s EMA-GhostConv proposal."""

import unittest
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import EMA, GhostConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


CFG = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml"
BASELINE_CFG = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/12/yolo12.yaml"
SEMANTIC_LAYER_MAP = {**{index: index for index in range(15)}, 16: 17, 17: 18, 19: 21, 20: 22, 21: 23}


class TestYOLO12EMAGhost(unittest.TestCase):
    """Focused structural tests for the proposed model."""

    @staticmethod
    def _baseline_model():
        """Build the unmodified YOLO12s graph for deterministic semantic-transfer tests."""
        cfg = YAML.load(BASELINE_CFG)
        cfg["scale"] = "s"
        cfg["yaml_file"] = "yolo12s.yaml"
        model = DetectionModel(cfg, nc=80, verbose=False).eval()
        with torch.no_grad():
            for index, value in enumerate(model.state_dict().values(), start=1):
                value.fill_(float(index) if value.is_floating_point() else index)
        return model

    def test_ema_preserves_shape_and_backpropagates(self):
        """EMA preserves P3/P4 feature dimensions and produces finite input gradients."""
        for channels, shape in ((128, (2, 128, 8, 8)), (256, (2, 256, 4, 4))):
            with self.subTest(channels=channels):
                x = torch.randn(shape, requires_grad=True)
                y = EMA(channels, factor=32)(x)

                self.assertEqual(y.shape, x.shape)
                y.mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_ema_rejects_invalid_channel_groups(self):
        """EMA rejects unsupported channel/factor combinations at model construction time."""
        for channels, factor in ((16, 32), (130, 32), (128, 0)):
            with self.subTest(channels=channels, factor=factor), self.assertRaisesRegex(ValueError, "EMA requires"):
                EMA(channels, factor=factor)

    def test_yolo12s_ema_ghost_graph_and_forward(self):
        """Build the final graph and verify the P3/P4/P5 detection-path tensor geometry."""
        model = DetectionModel(CFG, nc=5, verbose=False).eval()

        self.assertIsInstance(model.model[15], EMA)
        self.assertIsInstance(model.model[16], GhostConv)
        self.assertIsInstance(model.model[19], EMA)
        self.assertIsInstance(model.model[20], GhostConv)
        self.assertEqual(model.model[-1].f, [15, 19, 22])
        self.assertEqual(model.model[-1].nc, 5)

        shapes = {}
        hooks = [
            model.model[index].register_forward_hook(
                lambda _, __, output, index=index: shapes.__setitem__(index, tuple(output.shape))
            )
            for index in (15, 16, 17, 18, 19, 20, 21, 22)
        ]
        try:
            with torch.inference_mode():
                prediction = model(torch.zeros(1, 3, 64, 64))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(
            shapes,
            {
                15: (1, 128, 8, 8),
                16: (1, 128, 4, 4),
                17: (1, 384, 4, 4),
                18: (1, 256, 4, 4),
                19: (1, 256, 4, 4),
                20: (1, 256, 2, 2),
                21: (1, 768, 2, 2),
                22: (1, 512, 2, 2),
            },
        )
        self.assertEqual(prediction[0].shape, (1, 9, 84))

    def test_semantic_pretrained_transfer_preserves_shifted_modules(self):
        """Transfer every unchanged YOLO12s module despite the proposed graph's shifted layer indices."""
        source = self._baseline_model()
        model = YOLO(CFG)
        target_before = {key: value.clone() for key, value in model.model.state_dict().items()}

        model.load(source)
        report = model.pretrained_transfer_report
        target_state = model.model.state_dict()
        transferred = 0
        transferred_keys = set()
        for source_key, source_value in source.state_dict().items():
            _, source_index, suffix = source_key.split(".", 2)
            target_index = SEMANTIC_LAYER_MAP.get(int(source_index))
            if target_index is None:
                continue
            target_key = f"model.{target_index}.{suffix}"
            if target_key in target_state and source_value.shape == target_state[target_key].shape:
                self.assertTrue(torch.equal(source_value, target_state[target_key]), target_key)
                transferred += 1
                transferred_keys.add(target_key)

        self.assertEqual(transferred, 679)
        self.assertEqual(report["mode"], "semantic")
        self.assertEqual(report["exact_transferred_tensors"], 679)
        self.assertEqual(report["new_module_tensors"], 36)
        self.assertEqual(report["exact_transferred_parameter_tensors"], 346)
        self.assertEqual(report["exact_transferred_parameter_elements"], 8546048)
        self.assertTrue(
            all(
                torch.equal(target_state[key], value)
                for key, value in target_before.items()
                if key not in transferred_keys
            )
        )

    def test_semantic_transfer_keeps_five_class_logits_fresh(self):
        """Keep only class-logit outputs fresh when a COCO source is adapted to the five-class target."""
        source = self._baseline_model()
        target = DetectionModel(CFG, nc=5, verbose=False).eval()
        target_before = {key: value.clone() for key, value in target.state_dict().items()}

        report = target.load(source, verbose=False)
        target_state = target.state_dict()
        transferred_keys = set()
        for source_key, source_value in source.state_dict().items():
            _, source_index, suffix = source_key.split(".", 2)
            target_index = SEMANTIC_LAYER_MAP.get(int(source_index))
            if target_index is None:
                continue
            target_key = f"model.{target_index}.{suffix}"
            if target_key in target_state and source_value.shape == target_state[target_key].shape:
                transferred_keys.add(target_key)

        self.assertEqual(report["mode"], "semantic")
        self.assertEqual(report["exact_transferred_tensors"], 673)
        self.assertEqual(report["shape_mismatch_tensors"], 6)
        self.assertEqual(report["uninitialized_target_tensors"], 42)
        self.assertEqual(report["exact_transferred_parameter_tensors"], 340)
        self.assertEqual(report["exact_transferred_parameter_elements"], 8515088)
        self.assertTrue(
            all(
                torch.equal(target_state[key], value)
                for key, value in target_before.items()
                if key not in transferred_keys
            )
        )

    def test_proposed_checkpoint_uses_standard_key_shape_transfer(self):
        """Do not apply the baseline semantic map when loading an already-proposed checkpoint."""
        source = DetectionModel(CFG, nc=80, verbose=False).eval()
        target = DetectionModel(CFG, nc=80, verbose=False).eval()

        report = target.load(source, verbose=False)

        self.assertEqual(report["mode"], "key_shape")
        self.assertEqual(report["exact_transferred_tensors"], len(target.state_dict()))
