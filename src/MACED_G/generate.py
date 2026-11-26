import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from model import MACED_G


CELL_TYPES = ("K562", "HepG2", "SK-N-SH")


def build_model_config():
    return SimpleNamespace(
        d_intermediate=0,
        d_model=256,
        n_layer=8,
        fused_add_norm=True,
        initializer_range=0.02,
        pad_vocab_size_multiple=1,
        rescale_prenorm_residual=True,
        residual_in_fp32=True,
        rms_norm=True,
        ssm_cfg={"layer": "Mamba1"},
        vocab_size=7,
        tie_embeddings=True,
        attn_layer_idx=[],
        attn_cfg={},
        num_celltype=3,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(path: str):
    tokenizer_from_file = Tokenizer.from_file(path)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_from_file,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )


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
    model = MACED_G(build_model_config())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix(state_dict, "module.model.")
    state_dict = _strip_prefix(state_dict, "model.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


class MACEDGenerator:
    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bos_token_id = tokenizer.bos_token_id
        self.vocab_size = len(tokenizer)
        self._allowed_mask_cache = {}

    def _get_allowed_mask(self, allowed_tokens: Optional[List[str]]):
        if allowed_tokens is None:
            return None
        cache_key = tuple(sorted(allowed_tokens))
        if cache_key not in self._allowed_mask_cache:
            token_ids = []
            for token in allowed_tokens:
                token_ids.extend(self.tokenizer.encode(token, add_special_tokens=False))
            mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            mask[list(set(token_ids))] = True
            self._allowed_mask_cache[cache_key] = mask
        return self._allowed_mask_cache[cache_key]

    def _sample_inplace(self, logits, temperature, top_k, top_p, allowed_mask, out):
        if allowed_mask is not None:
            logits.masked_fill_(~allowed_mask, float("-inf"))
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if temperature != 1.0:
            logits.div_(temperature)
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits.masked_fill_(logits < values[..., -1:], float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = remove.scatter(1, sorted_indices, remove)
            logits.masked_fill_(remove, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        out.copy_(torch.multinomial(probs, num_samples=1).squeeze(-1))

    @torch.no_grad()
    def generate(
        self,
        activity: torch.Tensor,
        sequence_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 1.0,
        allowed_tokens: Optional[List[str]] = None,
        show_progress: bool = False,
    ):
        self.model.eval()
        activity = activity.to(self.device, non_blocking=True)
        batch_size = activity.shape[0]
        allowed_mask = self._get_allowed_mask(allowed_tokens)
        generated_ids = torch.empty(
            batch_size,
            sequence_length + 1,
            dtype=torch.long,
            device=self.device,
        )
        generated_ids[:, 0].fill_(self.bos_token_id)
        temp_logits = torch.empty(
            batch_size,
            self.vocab_size,
            dtype=torch.float32,
            device=self.device,
        )
        activity_emb = self.model.activity_encoder(activity)

        from mamba_ssm.utils.generation import InferenceParams

        inference_params = InferenceParams(
            max_batch_size=batch_size,
            max_seqlen=sequence_length + 1,
        )
        bos_emb = self.model.backbone.embedding(generated_ids[:, 0:1])
        hidden = self.model._forward_layers(
            bos_emb,
            activity_emb=activity_emb,
            inference_params=inference_params,
        )
        inference_params.seqlen_offset = 1
        temp_logits.copy_(self.model.lm_head(hidden)[:, 0, :])
        self._sample_inplace(
            temp_logits,
            temperature,
            top_k,
            top_p,
            allowed_mask,
            generated_ids[:, 1],
        )

        iterator = range(1, sequence_length)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating", unit="nt")
        single_token = torch.empty(batch_size, 1, dtype=torch.long, device=self.device)

        for position in iterator:
            single_token[:, 0] = generated_ids[:, position]
            token_emb = self.model.backbone.embedding(single_token)
            hidden = self.model._forward_layers(
                token_emb,
                activity_emb=activity_emb,
                inference_params=inference_params,
            )
            inference_params.seqlen_offset += 1
            temp_logits.copy_(self.model.lm_head(hidden)[:, 0, :])
            self._sample_inplace(
                temp_logits,
                temperature,
                top_k,
                top_p,
                allowed_mask,
                generated_ids[:, position + 1],
            )

        sequences = []
        for row in generated_ids[:, 1:]:
            sequence = self.tokenizer.decode(row.tolist(), skip_special_tokens=True).replace(" ", "")
            sequences.append(sequence)
        return sequences


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


def build_parser():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate 200-bp enhancer sequences with MACED-G")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .tsv, .csv, or .parquet")
    parser.add_argument("--activity", nargs=3, type=float, metavar=("K562", "HepG2", "SKNSH"), required=True)
    parser.add_argument("--num-sequences", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1215)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokenizer", default=str(here / "tokenizer.json"))
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_sequences <= 0:
        raise ValueError("num-sequences must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be > 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint, device)
    generator = MACEDGenerator(model, tokenizer, device)

    activity_vector = torch.tensor(args.activity, dtype=torch.float32)
    sequences = []
    remaining = args.num_sequences

    while remaining > 0:
        current_batch = min(args.batch_size, remaining)
        activity = activity_vector.unsqueeze(0).repeat(current_batch, 1)
        sequences.extend(
            generator.generate(
                activity=activity,
                sequence_length=args.sequence_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                allowed_tokens=["A", "C", "G", "T"],
                show_progress=args.show_progress,
            )
        )
        remaining -= current_batch

    output = pd.DataFrame({
        "sample_id": np.arange(len(sequences), dtype=np.int64),
        "sequence": sequences,
        "target_K562": args.activity[0],
        "target_HepG2": args.activity[1],
        "target_SKNSH": args.activity[2],
        "seed": args.seed
    })
    write_table(output, args.output)
    print(f"Saved {len(output)} sequences to {args.output}")


if __name__ == "__main__":
    main()
