import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

import dataset as myDataset
import model as myModel


def parse_devices(value):
    if value == "auto":
        return "auto"
    return [int(x) for x in value.split(",") if x.strip()]


def build_parser():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description="Train MACED-G")
    parser.add_argument("--train-data", default=str(root / "data" / "df_train.parquet"))
    parser.add_argument("--valid-data", default=str(root / "data" / "df_valid.parquet"))
    parser.add_argument("--output-dir", default=str(here / "train_output" / "PL_MACED_G"))
    parser.add_argument("--tokenizer", default=str(here / "tokenizer.json"))
    parser.add_argument("--devices", default="0", help="GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument("--strategy", default="auto", choices=["auto", "ddp"])
    parser.add_argument("--seed", type=int, default=1215)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--val-check-interval", type=int, default=10000)
    # parser.add_argument("--save-every-n-steps", type=int, default=10000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--valid-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser


class NanDetectionCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                if trainer.is_global_zero:
                    print(f"Non-finite gradient detected in {name}; stopping training.")
                trainer.should_stop = True
                return


def main():
    args = build_parser().parse_args()
    devices = parse_devices(args.devices)
    strategy = args.strategy
    if strategy == "auto" and isinstance(devices, list) and len(devices) > 1:
        strategy = "ddp"

    train_config = omegaconf.OmegaConf.create({
        "model_dir": args.output_dir,
        "lr": args.lr,
        "pct_start": args.pct_start,
        "devices": devices,
        "seed": args.seed,
        "trainer_save_top_n": 2,
        "max_steps": args.max_steps,
        "val_check_interval": args.val_check_interval,
        "valid_df_path": args.valid_data,
        "valid_batch_size": args.valid_batch_size,
        "train_df_path": args.train_data,
        "train_batch_size": args.train_batch_size,
        "activity_cols": ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"],
        "tokenizer_json": args.tokenizer,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "seq_col": "sequence",
    })
    model_config = omegaconf.OmegaConf.create({
        "d_intermediate": 0,
        "d_model": 256,
        "n_layer": 8,
        "fused_add_norm": True,
        "initializer_range": 0.02,
        "pad_vocab_size_multiple": 1,
        "rescale_prenorm_residual": True,
        "residual_in_fp32": True,
        "rms_norm": True,
        "ssm_cfg": {"layer": "Mamba1"},
        "vocab_size": 7,
        "tie_embeddings": True,
        "attn_layer_idx": [],
        "attn_cfg": {},
        "num_celltype": 3,
    })

    pl.seed_everything(train_config.seed, workers=True)
    tokenizer_from_file = Tokenizer.from_file(train_config.tokenizer_json)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_from_file,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )

    pl_model = myModel.PL_MACED_G(model_config, train_config)
    train_ds = myDataset.MPRA_Dataset(
        df_path=train_config.train_df_path,
        activity_cols=train_config.activity_cols,
        seq_col=train_config.seq_col,
        tokenizer=tokenizer,
        preload=args.preload,
    )
    valid_ds = myDataset.MPRA_Dataset(
        df_path=train_config.valid_df_path,
        activity_cols=train_config.activity_cols,
        seq_col=train_config.seq_col,
        tokenizer=tokenizer,
        preload=args.preload,
    )

    loader_kwargs = dict(
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=False,
        collate_fn=myDataset.collate_fn_MPRA_Dataset,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=train_config.valid_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    Path(train_config.model_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=train_config.model_dir)
    periodic_checkpoint = ModelCheckpoint(
        dirpath=train_config.model_dir,
        save_top_k=-1,
        monitor="step",
        filename="model-{step}",
        save_on_train_epoch_end=False,
        save_last=True
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=train_config.model_dir,
        save_top_k=train_config.trainer_save_top_n,
        monitor="valid_loss",
        filename="model-{step}-{valid_loss:.3f}",
        mode="min",
    )

    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[NanDetectionCallback(), periodic_checkpoint, best_checkpoint],
        max_steps=train_config.max_steps,
        val_check_interval=train_config.val_check_interval,
        precision=precision,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None
    )
    trainer.fit(pl_model, train_loader, val_loader)
    if trainer.is_global_zero:
        print(f"Best checkpoint: {best_checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
