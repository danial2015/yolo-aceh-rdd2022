# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused structural tests for the YOLO12s EMA-GhostConv proposal."""

import unittest
from pathlib import Path

import torch

from ultralytics.nn.modules import EMA, GhostConv
from ultralytics.nn.tasks import DetectionModel


CFG = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml"


class TestYOLO12EMAGhost(unittest.TestCase):
    """Focused structural tests for the proposed model."""

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
