from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
ACGT_IDS = {"A": 3, "C": 4, "G": 5, "T": 6}


def _build_lookup(acgt_ids=ACGT_IDS):
    lookup = np.full(256, -1, dtype=np.int32)
    for base, token_id in acgt_ids.items():
        lookup[ord(base)] = token_id
    return lookup


def encode_sequences(
    sequences,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    pad_id=PAD_ID,
    acgt_ids=ACGT_IDS,
    out_dtype=np.int64,
):
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 2), dtype=out_dtype)

    lookup = _build_lookup(acgt_ids)
    lengths = np.fromiter((len(sequence) for sequence in sequences), dtype=np.int64, count=n)
    length = int(lengths[0])

    if bool((lengths == length).all()):
        buffer = "".join(sequences).encode("ascii", errors="replace")
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(n, length)
        ids = lookup[array]
        if bool((ids >= 0).all()):
            output = np.empty((n, length + 2), dtype=out_dtype)
            output[:, 0] = bos_id
            output[:, 1:1 + length] = ids
            output[:, -1] = eos_id
            return output

    encoded = []
    max_length = 0
    for sequence in sequences:
        bytes_array = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
        row = lookup[bytes_array]
        row = row[row >= 0]
        full = np.empty(len(row) + 2, dtype=out_dtype)
        full[0] = bos_id
        full[1:-1] = row
        full[-1] = eos_id
        encoded.append(full)
        max_length = max(max_length, len(full))

    output = np.full((n, max_length), pad_id, dtype=out_dtype)
    for index, row in enumerate(encoded):
        output[index, :len(row)] = row
    return output


@dataclass
class Batch:
    seq_ids: torch.Tensor
    activity: torch.Tensor


class MPRA_Dataset(Dataset):
    def __init__(
        self,
        df_path,
        activity_cols,
        tokenizer=None,
        seq_col="sequence",
        bos="<|bos|>",
        eos="<|eos|>",
        pad="<|pad|>",
        preload=False,
    ):
        if df_path.endswith(".parquet"):
            df = pd.read_parquet(df_path)
        elif df_path.endswith(".csv"):
            df = pd.read_csv(df_path)
        elif df_path.endswith(".tsv"):
            df = pd.read_csv(df_path, sep="\t")
        else:
            raise ValueError(f"Unsupported format: {df_path}")

        self.preload = preload
        self.n_samples = len(df)
        self.activity = df[activity_cols].to_numpy(dtype=np.float32)

        if tokenizer is not None:
            vocab = tokenizer.get_vocab()
            self.bos_id = vocab[bos]
            self.eos_id = vocab[eos]
            self.pad_id = vocab[pad]
            self.acgt_ids = {base: vocab[base] for base in "ACGT"}
        else:
            self.bos_id = BOS_ID
            self.eos_id = EOS_ID
            self.pad_id = PAD_ID
            self.acgt_ids = dict(ACGT_IDS)

        sequences = df[seq_col].astype(str).str.upper().tolist()
        if preload:
            self.seq_ids = encode_sequences(
                sequences,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
                acgt_ids=self.acgt_ids,
            )
            self.sequences = None
        else:
            self.sequences = sequences
            self.seq_ids = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.preload:
            return self.seq_ids[index], self.activity[index]
        return self.sequences[index], self.activity[index]


def collate_fn_MPRA_Dataset(items):
    sequences, activities = zip(*items)
    activities = np.stack(activities, axis=0)
    if isinstance(sequences[0], str):
        seq_ids = encode_sequences(list(sequences))
    else:
        seq_ids = np.stack(sequences, axis=0)
    return Batch(
        seq_ids=torch.from_numpy(seq_ids),
        activity=torch.from_numpy(activities),
    )
