import torch
from torch import nn
from torchmetrics import MeanMetric, PearsonCorrCoef


class RegressionMetrics(nn.Module):
    """Streaming DDP-safe regression metrics for [batch, num_celltype]."""

    def __init__(self, num_celltype: int):
        super().__init__()
        self.pearson = PearsonCorrCoef(num_outputs=num_celltype)
        self.mse = MeanMetric()
        self.num_celltype = num_celltype
        self.register_buffer("n_samples", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor, loss: torch.Tensor):
        preds = preds.detach().float()
        target = target.detach().float()
        if preds.shape != target.shape or preds.ndim != 2:
            raise ValueError(f"Expected matching [batch, celltype] tensors, got {preds.shape} and {target.shape}")
        if preds.shape[1] != self.num_celltype:
            raise ValueError(f"Expected {self.num_celltype} cell types; got {preds.shape[1]}")
        self.pearson.update(preds, target)
        self.mse.update(loss.detach().float(), weight=preds.shape[0])
        self.n_samples += preds.shape[0]

    def compute(self):
        # PearsonCorrCoef / MeanMetric perform DDP synchronization on compute().
        per_cell = self.pearson.compute()
        mean_pearson = torch.nanmean(per_cell)
        return {
            "loss": self.mse.compute(),
            "pearson_per_cell": per_cell,
            "pearson_mean": mean_pearson,
            "pearson_sum": torch.nansum(per_cell),
        }

    def reset(self):
        self.pearson.reset()
        self.mse.reset()
        self.n_samples.zero_()
