import torch
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class SequenceMetrics(nn.Module):
    """Streaming token-level metrics for next-token prediction. DDP-safe."""

    def __init__(self, vocab_size: int):
        super().__init__()
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4 for top-4 accuracy")
        self.loss_seq_logits = MeanMetric()
        self.loss = MeanMetric()
        self.acc_seq = MulticlassAccuracy(num_classes=vocab_size, top_k=1, average="micro")
        self.acc_seq_top2 = MulticlassAccuracy(num_classes=vocab_size, top_k=2, average="micro")
        self.acc_seq_top4 = MulticlassAccuracy(num_classes=vocab_size, top_k=4, average="micro")

    @torch.no_grad()
    def update(self, seq_logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor):
        # seq_logits: [B, L, V], targets: [B, L]
        if seq_logits.ndim != 3 or targets.ndim != 2:
            raise ValueError("Expected seq_logits [B, L, V] and targets [B, L]")
        if seq_logits.shape[:2] != targets.shape:
            raise ValueError(f"Mismatched shapes: {seq_logits.shape} vs {targets.shape}")
        n_tokens = targets.numel()
        self.loss_seq_logits.update(loss.detach(), weight=n_tokens)
        self.loss.update(loss.detach(), weight=n_tokens)
        logits = seq_logits.detach().permute(0, 2, 1).contiguous()
        targets = targets.detach()
        self.acc_seq.update(logits, targets)
        self.acc_seq_top2.update(logits, targets)
        self.acc_seq_top4.update(logits, targets)

    def compute(self, prefix: str):
        return {
            f"{prefix}_loss_seq_logits": self.loss_seq_logits.compute(),
            f"{prefix}_loss": self.loss.compute(),
            f"{prefix}_acc_seq": self.acc_seq.compute(),
            f"{prefix}_acc_seq_top2": self.acc_seq_top2.compute(),
            f"{prefix}_acc_seq_top4": self.acc_seq_top4.compute(),
        }
    def reset(self):
        self.loss_seq_logits.reset()
        self.loss.reset()
        self.acc_seq.reset()
        self.acc_seq_top2.reset()
        self.acc_seq_top4.reset()
