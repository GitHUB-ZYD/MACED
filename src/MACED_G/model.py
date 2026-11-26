import copy
import math
from collections import defaultdict, namedtuple
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from metrics import SequenceMetrics
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError as exc:
    raise ImportError("MACED requires the fused Mamba RMSNorm kernels") from exc

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
                                               
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


                                                                                                                                              
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,                                      
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,                              
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:                                                                                             
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:                                             
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,                    
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
                                                                     
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MACED_G(nn.Module):
    def __init__(
        self,
        config,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        self.activity_encoder = nn.Sequential(
            nn.Linear(config.num_celltype, config.d_model),
            nn.LayerNorm(config.d_model),                     
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),          
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def _forward_layers(self, hidden_states, activity_emb=None, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.backbone.layers:
                                               
            if activity_emb is not None:
                                      
                hidden_states = hidden_states + 0.1 * activity_emb.unsqueeze(1)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.backbone.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.backbone.norm_f.weight,
                self.backbone.norm_f.bias,
                eps=self.backbone.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.backbone.residual_in_fp32,
                is_rms_norm=isinstance(self.backbone.norm_f, RMSNorm)
            )
        return hidden_states

    def forward(self, activity, seq_ids, inference_params=None, **mixer_kwargs):
                    
        activity_emb = self.activity_encoder(activity)                               
        seq_emb = self.backbone.embedding(seq_ids)                                                               
        h_all = self._forward_layers(seq_emb, activity_emb=activity_emb, inference_params=inference_params, **mixer_kwargs)                 
        h_for_pred = h_all[:, :-1, :]                               
        seq_logits = self.lm_head(h_for_pred)                                  
        CustomOutput = namedtuple("CustomOutput", ["seq_logits"])
        return CustomOutput(seq_logits=seq_logits)

class PL_MACED_G(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.model = MACED_G(model_config)
        self.model_config = model_config
        self.train_config = train_config
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (self.n_params / 1e6,))

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train_metrics = SequenceMetrics(vocab_size=model_config.vocab_size)
        self.valid_metrics = SequenceMetrics(vocab_size=model_config.vocab_size)

    def _common_step(self, batch, metrics):
        seq_ids = batch.seq_ids
        regression = batch.activity.float()
        output = self.model(activity=regression, seq_ids=seq_ids)
        seq_logits = output.seq_logits                   # [B, L-1, V]
        targets = seq_ids[:, 1:].long()                  # [B, L-1]

        loss = self.cross_entropy_loss(
            seq_logits.permute(0, 2, 1), targets
        )
        metrics.update(seq_logits, targets, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, self.valid_metrics)

    def on_validation_epoch_end(self):
        # compute() synchronizes TorchMetrics states across DDP ranks by default.
        log_metric_dict = {}
        log_metric_dict.update(self.train_metrics.compute("train"))
        log_metric_dict.update(self.valid_metrics.compute("valid"))

        try:
            lr = self.lr_schedulers().get_last_lr()[0]
        except (AttributeError, TypeError, IndexError):
            lr = self.train_config.lr

        log_metric_dict["lr"] = float(lr)
        # global_step is already recorded by Lightning loggers; log separately if desired.
        log_metric_dict["step"] = float(self.global_step)

        # All ranks have the same synced results; do not all_reduce a second time.
        self.log_dict(log_metric_dict, on_step=False, on_epoch=True,
                      prog_bar=False, logger=True, sync_dist=False, batch_size=1)
        self.train_metrics.reset()
        self.valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.train_config.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.train_config.pct_start,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
