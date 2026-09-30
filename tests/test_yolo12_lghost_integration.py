"""CPU smoke test of actual trainer, interrupted resume, validation and compact checkpoint reload.

Synthetic images only: this verifies execution, never research accuracy. Run explicitly with unittest.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ultralytics import YOLO
from ultralytics.models.yolo.detect.lghost import LGhostDetectionTrainer
from ultralytics.nn.modules.lghost import LearnedGroupConv, compact_lghost
from ultralytics.utils import YAML


class InterruptedSmokeRun(Exception):
    """Intentional interruption after a resumable checkpoint is saved."""


class TestLGhostIntegration(unittest.TestCase):
    """Exercise the public YOLO entrypoint and checkpoint serialization."""

    def test_train_resume_and_compact_reload(self):
        """Resume the original five-epoch schedule after pruning, then reload a compact YOLO checkpoint."""
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="lghost-smoke-") as temp:
            tmp = Path(temp)
            for split in ("train", "val", "test"):
                for name in ("images", "labels"):
                    (tmp / split / name).mkdir(parents=True)
                for i in range(4):
                    image = Image.new("RGB", (64, 64), (35 + i * 15, 40, 45))
                    ImageDraw.Draw(image).rectangle((20, 20, 44, 44), fill=(220, 190, 130))
                    image.save(tmp / split / "images" / f"{i}.jpg")
                    (tmp / split / "labels" / f"{i}.txt").write_text(f"{i % 5} 0.5 0.5 0.375 0.375\n")
            data = tmp / "data.yaml"
            YAML.save(
                data,
                dict(
                    path=str(tmp),
                    train="train/images",
                    val="val/images",
                    test="test/images",
                    names={0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "Repair"},
                ),
            )
            model = YOLO(root / "ultralytics/cfg/models/12/yolo12s-lghost-ema32.yaml")

            def interrupt(trainer):
                if trainer.epoch == 1:
                    raise InterruptedSmokeRun()

            model.add_callback("on_model_save", interrupt)
            # Execution smoke uses random initialization, independently of semantic pretrained regression tests.
            with self.assertRaises(InterruptedSmokeRun):
                model.train(
                    trainer=LGhostDetectionTrainer,
                    data=str(data),
                    epochs=5,
                    imgsz=64,
                    batch=2,
                    nbs=2,
                    device="cpu",
                    workers=0,
                    pretrained=False,
                    optimizer="SGD",
                    lr0=0.001,
                    patience=0,
                    amp=False,
                    plots=False,
                    cache=False,
                    project=str(tmp / "runs"),
                    name="smoke",
                    close_mosaic=0,
                    mosaic=0,
                    scale=0,
                    translate=0,
                    fliplr=0,
                    warmup_epochs=0,
                    save=True,
                )
            last = model.trainer.last
            resumed = YOLO(last)
            self.assertEqual([int(m.stage) for m in resumed.model.modules() if isinstance(m, LearnedGroupConv)], [1, 1])
            resumed.train(trainer=LGhostDetectionTrainer, resume=True, device="cpu")
            final = YOLO(resumed.trainer.last)
            blocks = [m for m in final.model.modules() if isinstance(m, LearnedGroupConv)]
            self.assertEqual([int(m.stage) for m in blocks], [3, 3])
            for m in blocks:
                self.assertTrue((m.mask.sum(1) == m.conv.in_channels // m.groups).all())
            final.model = compact_lghost(final.model).float().eval()
            compact_path = tmp / "compact.pt"
            final.save(compact_path)
            loaded = YOLO(compact_path)
            with torch.no_grad():
                x = torch.rand(1, 3, 64, 64)
                # YOLO.save stores FP16: compare against the saved rounded weights, not the FP32 original.
                expected = final.model.half().float()(x)[0]
                actual = loaded.model(x)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            loaded.val(
                data=str(data),
                split="test",
                imgsz=64,
                batch=2,
                device="cpu",
                workers=0,
                plots=False,
                project=str(tmp / "runs"),
                name="compact-test",
            )


if __name__ == "__main__":
    unittest.main()
