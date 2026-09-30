"""High-risk regression coverage: pruning, checkpoints, semantic transfer and detection gradients."""

import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.models.yolo.detect.lghost import pruning_stage, update_pruning
from ultralytics.nn.modules import EMA
from ultralytics.nn.modules.lghost import LGhostBottleneck, LearnedGroupConv, compact_lghost
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace, YAML
from ultralytics.utils.torch_utils import ModelEMA


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "ultralytics/cfg/models/12/yolo12s-lghost-ema32.yaml"


class TestLGhost(unittest.TestCase):
    """Keep old graph behavior unchanged while validating the new experimental branch."""

    def test_mask_gradient_optimizer_and_compaction(self):
        """Pruned columns cannot regrow through optimizer momentum; compact inference is equivalent."""
        torch.manual_seed(42)
        module = LearnedGroupConv(16, 24, 4)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        x = torch.randn(2, 16, 7, 9)
        module(x).square().mean().backward()
        optimizer.step()
        for stage in (1, 2, 3):
            previous = module.mask.clone()
            module.prune_to_stage(stage)
            self.assertTrue((~module.mask | previous).all())
            self.assertTrue((module.mask.sum(1) == 16 - stage * 4).all())
            optimizer.zero_grad()
            module(x).square().mean().backward()
            grad = module.conv.weight.grad.view(4, 6, 16)
            self.assertEqual(grad.masked_select(~module.mask[:, None, :]).abs().sum().item(), 0)
            self.assertGreater(grad.abs().sum().item(), 0)
            optimizer.step()
        module.eval()
        compact = module.to_compact()
        torch.testing.assert_close(module(x), compact(x), rtol=1e-5, atol=1e-6)
        self.assertEqual(compact.conv.weight.numel() * 4, module.conv.weight.numel())
        with self.assertRaises(ValueError):
            module.prune_to_stage(0)

    def test_schedule_checkpoint_and_weight_ema(self):
        """Pruning boundaries and discrete EMA masks survive a state-dict checkpoint round-trip."""
        model = torch.nn.Sequential(LearnedGroupConv(16, 16))
        model.yaml = {"lghost_pruning": {"ratio": 0.6}}
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SimpleNamespace(model=model, ema=ModelEMA(model), epochs=160, save_dir=Path(tmp))
            for epoch, expected in ((0, 0), (31, 0), (32, 1), (63, 1), (64, 2), (95, 2), (96, 3), (159, 3)):
                self.assertEqual(pruning_stage(epoch, 160, 4, 0.6), expected)
                trainer.epoch = epoch
                update_pruning(trainer)
                trainer.ema.update(model)
                self.assertTrue(torch.equal(model[0].mask, trainer.ema.ema[0].mask))
                self.assertEqual(int(trainer.ema.ema[0].stage), expected)
            stream = io.BytesIO()
            torch.save(trainer.ema.ema.state_dict(), stream)
            stream.seek(0)
            restored = deepcopy(model)
            restored.load_state_dict(torch.load(stream, weights_only=True))
            self.assertTrue(torch.equal(restored[0].mask, model[0].mask))
            with self.assertRaises(ValueError):
                pruning_stage(0, 160, 4, 1.0)

    def test_graph_pretrained_mapping_and_real_detection_backward(self):
        """Check unchanged modules transfer exactly and LG/EMA receive finite nonzero detection-loss gradients."""
        baseline_cfg = YAML.load(ROOT / "ultralytics/cfg/models/12/yolo12.yaml")
        baseline_cfg["scale"] = "s"
        source = DetectionModel(baseline_cfg, nc=80, verbose=False)
        target = DetectionModel(CFG, nc=5, verbose=False)
        new_weight = target.model[16].lg.conv.weight.detach().clone()
        report = target.load(source, verbose=False)
        self.assertEqual(report["mode"], "semantic")
        self.assertEqual(report["exact_transferred_tensors"], 673)
        self.assertEqual(report["exact_transferred_parameter_elements"], 8515088)
        torch.testing.assert_close(target.model[18].state_dict()["cv1.conv.weight"], source.model[17].cv1.conv.weight)
        torch.testing.assert_close(new_weight, target.model[16].lg.conv.weight)
        self.assertIsInstance(target.model[15], EMA)
        self.assertIsInstance(target.model[19], EMA)
        self.assertIsInstance(target.model[16], LGhostBottleneck)
        self.assertIsInstance(target.model[20], LGhostBottleneck)
        self.assertEqual(target.model[-1].f, [15, 19, 22])
        target.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.randn(2, 3, 64, 64),
            "batch_idx": torch.tensor([0.0, 1.0]),
            "cls": torch.tensor([[0.0], [4.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.3, 0.4, 0.2, 0.2]]),
        }
        loss, _ = target(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for index in (15, 16, 19, 20):
            grads = [p.grad for p in target.model[index].parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        target.eval()
        for module in target.modules():
            if isinstance(module, LearnedGroupConv):
                module.prune_to_stage(3)
        compact = compact_lghost(target)
        with torch.no_grad():
            expected = target(batch["img"])[0]
            actual = compact(batch["img"])[0]
        self.assertEqual(expected.shape, (2, 9, 84))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_geometry_and_incomplete_conversion(self):
        """Support unequal channels and odd spatial sizes without silently compacting dense blocks."""
        for stride in (1, 2):
            m = LGhostBottleneck(16, 24, s=stride).eval()
            y = m(torch.randn(2, 16, 9, 11))
            self.assertEqual(y.shape, (2, 24, (9 + stride - 1) // stride, (11 + stride - 1) // stride))
            with self.assertRaises(ValueError):
                compact_lghost(m)

    def test_flops_counts_dense_masked_convolution(self):
        """Count functional convolution once and retain dense cost even after pruning."""
        try:
            import thop
        except ImportError:
            self.skipTest("ultralytics-thop is not installed")
        from ultralytics.nn.modules.lghost import count_lghost

        m = LearnedGroupConv(16, 24).eval()
        x = torch.rand(1, 16, 8, 8)
        dense = thop.profile(m, inputs=[x], custom_ops={LearnedGroupConv: count_lghost}, verbose=False)[0]
        m.prune_to_stage(3)
        pruned = thop.profile(m, inputs=[x], custom_ops={LearnedGroupConv: count_lghost}, verbose=False)[0]
        compact = thop.profile(m.to_compact(), inputs=[x], verbose=False)[0]
        self.assertEqual(dense, pruned)
        self.assertEqual(dense - compact, 8 * 8 * 24 * (16 - 4))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
