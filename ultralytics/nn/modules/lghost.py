"""L-Ghost-inspired blocks (Chi et al., DOI: 10.1109/ACCESS.2023.3234108).

This is a detection adaptation: learned column pruning + Ghost projection, with EMA attention outside the block.
It does not implement the paper's p-CA or replace the YOLO backbone with L-GhostNet.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from .conv import Conv, DWConv, GhostConv


class LearnedGroupConv(nn.Module):
    """Dense pointwise training weights with monotonic, group-wise input-column pruning.

    Each output group independently retains input channels ranked by the L1 norm of its weights. At stage G-1,
    each group retains I/G inputs. Boolean masks and integer stages are checkpoint buffers, not trainable parameters.
    Masking remains in forward so optimizer momentum cannot reactivate pruned connections.
    """

    def __init__(self, c1: int, c2: int, groups: int = 4):
        """Initialize a masked pointwise convolution followed by BN and ReLU."""
        super().__init__()
        if groups < 2 or c1 % groups or c2 % groups:
            raise ValueError("LearnedGroupConv requires groups >= 2 and both channel counts divisible by groups")
        self.groups = groups
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)
        self.register_buffer("mask", torch.ones(groups, c1, dtype=torch.bool))
        self.register_buffer("stage", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        """Apply the masked kernel; gradients of all pruned weights are exactly zero."""
        mask = self.mask.repeat_interleave(self.conv.out_channels // self.groups, dim=0)[:, :, None, None]
        y = F.conv2d(x, self.conv.weight * mask)
        return self.act(self.bn(y))

    @torch.no_grad()
    def prune_to_stage(self, stage: int):
        """Prune one I/G-column tranche per stage, never restoring an eliminated column."""
        if not 0 <= stage < self.groups:
            raise ValueError("Pruning stage must be between 0 and groups - 1")
        if stage < int(self.stage):
            raise ValueError("Pruning cannot run backwards; resume with the original epoch budget")
        weight = self.conv.weight.view(self.groups, self.conv.out_channels // self.groups, self.conv.in_channels)
        for next_stage in range(int(self.stage) + 1, stage + 1):
            scores = weight.abs().sum(dim=1).masked_fill(~self.mask, float("inf"))
            # Stable sorting resolves ties consistently, including when continuing a checkpoint on another device.
            remove = scores.argsort(dim=1, stable=True)[:, : self.conv.in_channels // self.groups]
            self.mask.scatter_(1, remove, False)
            self.stage.fill_(next_stage)
        weight.mul_(self.mask[:, None, :])

    def to_compact(self):
        """Convert a fully pruned block to input gather + regular grouped convolution for inference."""
        if int(self.stage) != self.groups - 1:
            raise ValueError("Compact conversion requires the final pruning stage")
        return CompactLearnedGroupConv(self).train(self.training)


class CompactLearnedGroupConv(nn.Module):
    """Equivalent inference module with physically smaller weights and per-group gathered inputs.

    Input indices may repeat between groups: different learned groups may select the same input channel. The gather
    is necessary to preserve those learned connections, not a new learned channel-shuffle operation.
    """

    def __init__(self, source: LearnedGroupConv):
        """Copy selected columns and BN statistics without changing the function."""
        super().__init__()
        c1, c2, groups = source.conv.in_channels, source.conv.out_channels, source.groups
        indices = torch.stack([row.nonzero(as_tuple=True)[0] for row in source.mask])
        if indices.shape != (groups, c1 // groups):
            raise ValueError("Each learned group must retain exactly I/G inputs")
        self.register_buffer("indices", indices.flatten())
        self.conv = nn.Conv2d(c1, c2, 1, groups=groups, bias=False).to(source.conv.weight)
        weight = source.conv.weight.detach().view(groups, c2 // groups, c1)
        selected = torch.stack([weight[g].index_select(1, indices[g]) for g in range(groups)])
        with torch.no_grad():
            self.conv.weight.copy_(selected.reshape_as(self.conv.weight))
        self.bn = deepcopy(source.bn)
        self.act = deepcopy(source.act)

    def forward(self, x):
        """Gather independently learned inputs before the grouped pointwise convolution."""
        return self.act(self.bn(self.conv(x.index_select(1, self.indices))))


class LGhostBottleneck(nn.Module):
    """LG -> optional stride-2 DWConv -> Ghost projection + shortcut; no internal p-CA.

    Hidden width equals the output width by default (an explicit YOLO adaptation, not the paper's full Table 1).
    Existing EMA-32 layers remain outside this block at the original P3/P4 positions.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, groups: int = 4):
        """Initialize a stride-1/2 bottleneck with shape-compatible residual projection."""
        super().__init__()
        if s not in (1, 2) or k <= 0 or k % 2 == 0 or c2 % 2:
            raise ValueError("LGhostBottleneck requires stride 1/2, odd kernel and even output channels")
        self.lg = LearnedGroupConv(c1, c2, groups)
        self.dw = DWConv(c2, c2, k, s, act=False) if s == 2 else nn.Identity()
        self.project = GhostConv(c2, c2, 1, 1, act=False)
        self.shortcut = (
            nn.Identity()
            if s == 1 and c1 == c2
            else nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, act=False))
        )

    def forward(self, x):
        """Fuse the learned-group path and residual path."""
        return self.project(self.dw(self.lg(x))) + self.shortcut(x)


def compact_lghost(model: nn.Module):
    """Return a deep-copied inference model; refuse partially pruned or already compact models."""
    blocks = [m for m in model.modules() if isinstance(m, LearnedGroupConv)]
    if not blocks or any(int(m.stage) != m.groups - 1 for m in blocks):
        raise ValueError("All LearnedGroupConv blocks must be fully pruned before compact conversion")
    result = deepcopy(model).eval()
    for module in result.modules():
        if isinstance(module, LGhostBottleneck):
            module.lg = module.lg.to_compact()
    return result


def count_lghost(module, inputs, output):
    """THOP MAC count for the functional dense masked pointwise convolution.

    Pruned zeros do not reduce dense training MACs. Like standard THOP counts, mask/gather memory overhead and
    activation costs are excluded; measure latency separately instead of interpreting this estimate as FPS.
    THOP >= 2.1.6 also counts child modules, so BN is deliberately not counted again here.
    """
    module.total_ops += output.numel() * module.conv.in_channels
