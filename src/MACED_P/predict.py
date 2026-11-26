import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from dataset import encode_sequences
from model import MACED_P


PREDICTION_COLUMNS = [
    "pred_K562_log2FC",
    "pred_HepG2_log2FC",
    "pred_SKNSH_log2FC",
]


def build_model_config():
    return SimpleNamespace(
        d_intermediate=0,
        d_model=256,
        n_layer=8,
        fused_add_norm=True,
        initializer_range=0.1,
        pad_vocab_size_multiple=1,
        rescale_prenorm_residual=False,
        residual_in_fp32=True,
        rms_norm=True,
        ssm_cfg={"layer": "Mamba1"},
        vocab_size=7,
        tie_embeddings=True,
        attn_layer_idx=[],
        attn_cfg={},
        num_celltype=3,
    )


def read_table(path: str):
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".tsv":
        return pd.read_csv(input_path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError("Input must end with .tsv, .csv, or .parquet")


def write_table(df: pd.DataFrame, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".tsv":
        df.to_csv(output, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError("Output must end with .tsv, .csv, or .parquet")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def _strip_prefix(state_dict, prefix: str):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_model(checkpoint_path: str, device: torch.device):
    model = MACED_P(build_model_config())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix(state_dict, "module.model.")
    state_dict = _strip_prefix(state_dict, "model.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def validate_sequences(sequences, expected_length: int):
    invalid_length = []
    invalid_alphabet = []
    clean = []
    for index, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        clean.append(sequence)
        if len(sequence) != expected_length:
            invalid_length.append(index)
        if set(sequence) - set("ACGT"):
            invalid_alphabet.append(index)
    if invalid_length:
        preview = invalid_length[:10]
        raise ValueError(
            f"All sequences must be exactly {expected_length} bp. Invalid row indices: {preview}"
        )
    if invalid_alphabet:
        preview = invalid_alphabet[:10]
        raise ValueError(
            f"Sequences must contain only A/C/G/T. Invalid row indices: {preview}"
        )
    return clean


def build_parser():
    parser = argparse.ArgumentParser(description="Predict enhancer activity with MACED-P")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input .tsv, .csv, or .parquet")
    parser.add_argument("--output", required=True, help="Output .tsv, .csv, or .parquet")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    df = read_table(args.input)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Missing sequence column: {args.sequence_column}")

    sequences = validate_sequences(df[args.sequence_column].tolist(), args.sequence_length)
    seq_ids = encode_sequences(sequences)
    model = load_model(args.checkpoint, device)

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(seq_ids), args.batch_size):
            stop = min(start + args.batch_size, len(seq_ids))
            batch = torch.from_numpy(seq_ids[start:stop]).long().to(device, non_blocking=True)
            pred = model(seq_ids=batch)
            predictions.append(pred.float().cpu().numpy())

    if predictions:
        prediction_matrix = np.concatenate(predictions, axis=0)
    else:
        prediction_matrix = np.empty((0, 3), dtype=np.float32)

    output = df.copy()
    for column_index, column_name in enumerate(PREDICTION_COLUMNS):
        output[column_name] = prediction_matrix[:, column_index]

    write_table(output, args.output)
    print(f"Saved predictions for {len(output)} sequences to {args.output}")


if __name__ == "__main__":
    main()
