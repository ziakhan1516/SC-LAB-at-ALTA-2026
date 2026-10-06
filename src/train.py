#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALTA 2026 Shared Task -- Sentiment & Sarcasm classification for en-AU / en-UK.

Official metric:
    score = ( min(f1_sent_AU, f1_sent_UK) + min(f1_sarc_AU, f1_sarc_UK) ) / 2
(macro-F1 per task per variety; each task scored by its WEAKEST variety).

Strategy
--------
1. Multi-task encoder: one shared transformer, two heads (sentiment, sarcasm).
   The two tasks are jointly informative (sarcasm inverts surface sentiment),
   so sharing the encoder acts as regularisation for the tiny sarcasm signal.
2. Variety/source conditioning: the variety and source are prepended to the
   text so the model can learn variety-specific priors (en-UK/google is
   essentially never sarcastic: 1/1817 in train).
3. K-fold ensembling over several backbones/seeds, averaging probabilities.
4. Per-(variety, source) threshold tuning on out-of-fold probabilities,
   maximising the *variety-level* macro-F1 that the official metric uses.
   This is the main lever for the sarcasm score, which is heavily imbalanced.

Artifacts (out-of-fold + target probabilities) are cached per model/seed so
blends can be recomputed without retraining (`--blend_only`).
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
import random
import re
import sys
import time
import zipfile

# Must be set before torch initialises CUDA. On Windows the driver silently spills to
# shared system memory instead of raising OOM, which is ~3x slower and eventually crashes
# the process natively; expandable segments keep fragmentation (and therefore peak) down.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

VARIETIES = ["en-AU", "en-UK"]
TASKS = ["sentiment", "sarcasm"]


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    hf_name: str
    lr: float = 1e-5
    head_lr: float = 1e-4
    batch_size: int = 8
    accum: int = 2
    epochs: int = 5
    max_len: int = 192
    llrd: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # LLM backbones: LoRA adapters over a (optionally 4-bit) causal LM
    lora: bool = False
    load_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    grad_ckpt: bool = False


# Batch/length defaults are tuned for a 12 GB card. Measured peak allocation for
# deberta-v3-large on worst-case (longest) batches: 192/bs8 = 9.6 GiB @ 200 ms/step,
# 256/bs8 = 11.0 GiB @ 589 ms/step (memory-thrashing). Do not raise these blindly.
MODEL_ZOO: dict[str, ModelSpec] = {
    "deberta": ModelSpec("microsoft/deberta-v3-large", lr=8e-6, batch_size=8, accum=2, epochs=5),
    "deberta-base": ModelSpec("microsoft/deberta-v3-base", lr=2e-5, batch_size=16, accum=1, epochs=5),
    "roberta": ModelSpec("roberta-large", lr=1e-5, batch_size=8, accum=2, epochs=5),
    "modernbert": ModelSpec("answerdotai/ModernBERT-large", lr=1e-5, batch_size=8, accum=2, epochs=5),
    "distilbert": ModelSpec("distilbert-base-uncased", lr=3e-5, batch_size=32, accum=1, epochs=4, max_len=128),
    # LoRA lr is ~20x the full-fine-tuning lr: only the adapters train.
    "qwen8b": ModelSpec("Qwen/Qwen3-8B", lr=1e-4, head_lr=1e-3, batch_size=2, accum=8, epochs=3,
                        lora=True, load_4bit=True, grad_ckpt=True),
    # Qwen3 needs transformers>=4.51; this env is pinned to 4.46.3 by python 3.8. These two
    # 8B/7B decoders are already in the local HF cache and ARE supported by 4.46, so they are
    # the only way to add >4B capacity here -- the whole existing pool tops out at 4B.
    "llama8b": ModelSpec("/home/ali/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659", lr=1e-4, head_lr=1e-3,
                         batch_size=2, accum=8, epochs=3,
                         lora=True, load_4bit=True, grad_ckpt=True),
    "ds7b": ModelSpec("/home/ali/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60", lr=1e-4, head_lr=1e-3,
                      batch_size=2, accum=8, epochs=3,
                      lora=True, load_4bit=True, grad_ckpt=True),
    # from v2/alta_v2.py's zoo: both are supported by transformers 4.46 and were pool
    # members of the dev-phase ensemble (phi carried the largest sentiment-en-AU weight).
    "phi4mini": ModelSpec("microsoft/Phi-4-mini-instruct", lr=1e-4, head_lr=1e-3,
                          batch_size=4, accum=4, epochs=3,
                          lora=True, load_4bit=True, grad_ckpt=True),
    "mistral": ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", lr=1e-4, head_lr=1e-3,
                         batch_size=2, accum=8, epochs=3,
                         lora=True, load_4bit=True, grad_ckpt=True),
    "qwen4b": ModelSpec("Qwen/Qwen3-4B", lr=1e-4, head_lr=1e-3, batch_size=4, accum=4, epochs=3,
                        lora=True, load_4bit=True, grad_ckpt=True),
    "qwen1.7b": ModelSpec("Qwen/Qwen3-1.7B", lr=1e-4, head_lr=1e-3, batch_size=8, accum=2, epochs=3,
                          lora=True, load_4bit=False, grad_ckpt=True),
    # Small sibling of qwen4b: same architecture and code path, cheap enough to validate on.
    "qwen0.6b": ModelSpec("Qwen/Qwen3-0.6B", lr=1e-4, head_lr=1e-3, batch_size=8, accum=2, epochs=3,
                          lora=True, load_4bit=True, grad_ckpt=True),
}

# Causal LMs need last-token pooling rather than mean pooling.
DECODER_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "phi3", "gpt_neox"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def clean_text(t: str) -> str:
    """Light normalisation only -- sarcasm cues live in punctuation and casing."""
    t = html.unescape(str(t))
    t = t.replace(" ", " ")
    t = _WS.sub(" ", t).strip()
    return t if t else "empty"


def build_input(row, neutral: bool = False) -> str:
    """Condition the encoder on the variety and the source domain.

    neutral=True replaces the variety with a generic token (variety-invariance
    augmentation); the source is kept because en-UK/google is genuinely never sarcastic.
    """
    var = "en" if neutral else row["variety"]
    return f"variety: {var} | source: {row['source']} | {clean_text(row['text'])}"


class TextDataset(Dataset):
    """Pre-tokenised once (not per epoch) with head+tail truncation.

    Plain head truncation drops the end of long reviews, which is exactly where the
    verdict usually sits ("...overall, wouldn't go back"). Keeping the first 3/4 and
    last 1/4 recovers most of what a 192-token budget would otherwise lose.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, with_labels: bool,
                 tail_frac: float = 0.25, variety_aug: float = 0.0):
        """variety_aug>0 pre-tokenises a variety-neutral copy of every row and swaps it in
        with that probability at train time.

        en-UK has 218 sarcastic training rows against en-AU's 739. Conditioning on the
        variety token lets the encoder fit two separate detectors, so the UK cell -- which
        the min() metric scores -- learns from a third of the data. Randomly neutralising
        the token forces one shared, variety-invariant sarcasm detector.
        """
        budget = max_len - tokenizer.num_special_tokens_to_add(pair=False)
        n_tail = int(budget * tail_frac)
        n_head = budget - n_tail

        def encode(txts):
            bodies = tokenizer(txts, add_special_tokens=False, truncation=False)["input_ids"]
            out = []
            for ids in bodies:
                if len(ids) > budget:
                    ids = ids[:n_head] + ids[-n_tail:]
                out.append(tokenizer.build_inputs_with_special_tokens(ids))
            return out

        self.input_ids = encode([build_input(r) for _, r in df.iterrows()])
        self.variety_aug = variety_aug
        self.alt_ids = encode([build_input(r, neutral=True) for _, r in df.iterrows()]) \
            if variety_aug > 0 else None
        self.lengths = np.array([len(x) for x in self.input_ids])

        self.with_labels = with_labels
        if with_labels:
            self.y_sent = df["sentiment"].to_numpy(dtype=np.int64)
            self.y_sarc = df["sarcasm"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int):
        ids = self.input_ids[i]
        if self.alt_ids is not None and np.random.rand() < self.variety_aug:
            ids = self.alt_ids[i]
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }
        if self.with_labels:
            item["y_sent"] = torch.tensor(self.y_sent[i], dtype=torch.long)
            item["y_sarc"] = torch.tensor(self.y_sarc[i], dtype=torch.long)
        return item


def load_external_sarcasm(cache: Path) -> pd.DataFrame:
    """Public sarcasm/irony corpora for intermediate-task fine-tuning.

    Intermediate-task transfer is well supported for sarcasm (e.g. Chen et al., Mathematics
    2022, 'Intermediate-Task Transfer Learning with BERT for Sarcasm Detection'). BESSTIE is
    deliberately NOT used as a source: it would contain the gold labels of the shared task's
    test data, which the rules forbid.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)
    from datasets import load_dataset
    frames = []
    try:
        d = load_dataset("raquiba/Sarcasm_News_Headline")
        for split in d:
            df = d[split].to_pandas()
            frames.append(pd.DataFrame({"text": df["headline"].astype(str),
                                        "sarcasm": df["is_sarcastic"].astype(int)}))
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  [warn] Sarcasm_News_Headline unavailable: {e}")
    try:
        d = load_dataset("cardiffnlp/tweet_eval", "irony")
        for split in d:
            df = d[split].to_pandas()
            frames.append(pd.DataFrame({"text": df["text"].astype(str),
                                        "sarcasm": df["label"].astype(int)}))
    except Exception as e:  # pragma: no cover
        print(f"  [warn] tweet_eval/irony unavailable: {e}")
    if not frames:
        raise SystemExit("no external sarcasm corpus could be loaded")
    out = pd.concat(frames, ignore_index=True).dropna().drop_duplicates(subset="text")
    out = out[out.text.str.split().str.len() >= 3].reset_index(drop=True)
    out.to_csv(cache, index=False)
    return out


class ExternalDataset(Dataset):
    """Single-task (sarcasm-only) dataset for the intermediate stage."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        texts = [f"variety: en-UK | source: reddit | {clean_text(t)}" for t in df["text"]]
        budget = max_len - tokenizer.num_special_tokens_to_add(pair=False)
        bodies = tokenizer(texts, add_special_tokens=False, truncation=True,
                           max_length=budget)["input_ids"]
        self.input_ids = [tokenizer.build_inputs_with_special_tokens(b) for b in bodies]
        self.lengths = np.array([len(x) for x in self.input_ids])
        self.y = df["sarcasm"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids = self.input_ids[i]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "y_sent": torch.tensor(0, dtype=torch.long),
                "y_sarc": torch.tensor(self.y[i], dtype=torch.long)}


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, then sort by length inside large chunks so batches pad to similar lengths.

    Median input is ~62 tokens against a 192 budget, so random batching wastes most of
    every batch on padding. Batch order is reshuffled each epoch to keep training stochastic.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, chunk_mult: int = 50, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk = batch_size * chunk_mult
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths))
        chunks = [idx[i:i + self.chunk] for i in range(0, len(idx), self.chunk)]
        ordered = np.concatenate([c[np.argsort(self.lengths[c], kind="stable")] for c in chunks])
        batches = [ordered[i:i + self.batch_size] for i in range(0, len(ordered), self.batch_size)]
        rng.shuffle(batches)
        return iter(np.concatenate(batches).tolist())


class Collator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        labels = {}
        for key in ("y_sent", "y_sarc"):
            if key in batch[0]:
                labels[key] = torch.stack([b.pop(key) for b in batch])
        out = self.tok.pad(batch, return_tensors="pt")
        out.update(labels)
        return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultiTaskClassifier(nn.Module):
    """Shared encoder + mean pooling + two binary heads with multi-sample dropout."""

    def __init__(self, hf_name: str, n_msd: int = 5, p_drop: float = 0.2, spec: "ModelSpec | None" = None):
        super().__init__()
        spec = spec or ModelSpec(hf_name)
        cfg = AutoConfig.from_pretrained(hf_name)
        self.is_decoder = cfg.model_type in DECODER_TYPES

        if not self.is_decoder:
            cfg.update({"hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1})

        kwargs = {}
        if spec.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            # `dtype=` is the transformers>=4.56 spelling; 4.46 (pinned by python 3.8)
            # still calls it `torch_dtype`. Pick whichever this version accepts.
            import inspect as _insp
            _sig = _insp.signature(AutoModel.from_pretrained).parameters
            kwargs["dtype" if "dtype" in _sig else "torch_dtype"] = torch.bfloat16
        self.encoder = AutoModel.from_pretrained(hf_name, config=cfg, **kwargs)

        if spec.grad_ckpt:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = False

        self.lora = spec.lora
        if spec.lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if spec.load_4bit:
                self.encoder = prepare_model_for_kbit_training(
                    self.encoder, use_gradient_checkpointing=spec.grad_ckpt)
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=spec.lora_r, lora_alpha=spec.lora_alpha, lora_dropout=0.05, bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ))
            self.encoder.print_trainable_parameters()

        hidden = getattr(cfg, "hidden_size", None) or cfg.d_model
        self.dropouts = nn.ModuleList([nn.Dropout(p_drop) for _ in range(n_msd)])
        self.sent_head = nn.Linear(hidden, 2)
        self.sarc_head = nn.Linear(hidden, 2)
        for head in (self.sent_head, self.sarc_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, **inputs):
        inputs.pop("y_sent", None)
        inputs.pop("y_sarc", None)
        mask = inputs["attention_mask"]
        hidden = self.encoder(**inputs).last_hidden_state
        if self.is_decoder:
            # Causal attention: only the final real token has seen the whole sequence.
            # Inputs are right-padded, so that is index (n_tokens - 1).
            last = mask.sum(1).long() - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        else:
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        pooled = pooled.float()
        sent = torch.stack([self.sent_head(d(pooled)) for d in self.dropouts]).mean(0)
        sarc = torch.stack([self.sarc_head(d(pooled)) for d in self.dropouts]).mean(0)
        return sent, sarc


def _find_layer_list(encoder: nn.Module):
    for path in ("encoder.layer", "layers", "encoder.layers", "transformer.layer", "transformer.h"):
        obj = encoder
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    return None


def build_optimizer(model: MultiTaskClassifier, spec: ModelSpec):
    """AdamW with layer-wise learning-rate decay (stabilises *-large fine-tuning)."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")

    def group(params, lr):
        decay = [p for n, p in params if p.requires_grad and not any(k in n for k in no_decay)]
        plain = [p for n, p in params if p.requires_grad and any(k in n for k in no_decay)]
        out = []
        if decay:
            out.append({"params": decay, "lr": lr, "weight_decay": spec.weight_decay})
        if plain:
            out.append({"params": plain, "lr": lr, "weight_decay": 0.0})
        return out

    layers = None if spec.lora else _find_layer_list(model.encoder)
    groups, seen = [], set()
    if spec.lora:
        # Only the LoRA adapters train, so layer-wise decay has nothing to decay over.
        groups += group(list(model.encoder.named_parameters()), spec.lr)
    elif layers is not None:
        n = len(layers)
        for i, layer in enumerate(layers):
            lr = spec.lr * (spec.llrd ** (n - 1 - i))
            named = list(layer.named_parameters())
            seen.update(id(p) for _, p in named)
            groups += group(named, lr)
        rest = [(n_, p) for n_, p in model.encoder.named_parameters() if id(p) not in seen]
        groups += group(rest, spec.lr * (spec.llrd ** n))
    else:
        groups += group(list(model.encoder.named_parameters()), spec.lr)

    heads = list(model.sent_head.named_parameters()) + list(model.sarc_head.named_parameters())
    groups += group(heads, spec.head_lr)

    try:
        return torch.optim.AdamW(groups, eps=1e-6, betas=(0.9, 0.999), fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, eps=1e-6, betas=(0.9, 0.999))


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def official_score(df: pd.DataFrame, pred_sent: np.ndarray, pred_sarc: np.ndarray) -> tuple[dict, float]:
    scores = {}
    preds = {"sentiment": pred_sent, "sarcasm": pred_sarc}
    for task in TASKS:
        for var in VARIETIES:
            m = (df["variety"] == var).to_numpy()
            scores[f"{task}-{var}"] = f1_score(df.loc[m, task].to_numpy(), preds[task][m], average="macro")
    final = (
        min(scores[f"sentiment-{v}"] for v in VARIETIES)
        + min(scores[f"sarcasm-{v}"] for v in VARIETIES)
    ) / 2
    return scores, final


GRID = np.round(np.arange(0.05, 0.96, 0.01), 4)


def _f1_curve(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([f1_score(y, (p >= t).astype(int), average="macro") for t in GRID])


def bootstrap_threshold(y: np.ndarray, p: np.ndarray, n_boot: int = 60, seed: int = 0) -> float:
    """Median of the argmax over bootstrap resamples.

    A raw argmax over the F1-vs-threshold curve latches onto a single noisy spike and
    transfers badly. Measured on the deberta OOF/target pair: argmax 0.8371 vs
    bootstrap 0.8469. Costs nothing at run time.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        picks.append(GRID[int(np.argmax(_f1_curve(y[idx], p[idx])))])
    return float(np.median(picks)) if picks else 0.5


def tune_thresholds(prob: np.ndarray, y: np.ndarray, variety: np.ndarray, source: np.ndarray,
                    per_source: bool = True, min_gain: float = 0.002) -> dict[tuple[str, str], float]:
    """Threshold per (variety, source), chosen to maximise each variety's macro-F1.

    The official metric computes macro-F1 over a whole variety, so per-source
    thresholds are optimised jointly by coordinate ascent on the variety score.
    """
    thresholds: dict[tuple[str, str], float] = {}
    for var in np.unique(variety):
        vm = variety == var
        yv, pv, sv = y[vm], prob[vm], source[vm]

        best_t = bootstrap_threshold(yv, pv)
        best_f1 = f1_score(yv, (pv >= best_t).astype(int), average="macro")

        cur = {s: best_t for s in np.unique(sv)}
        if per_source and len(cur) > 1:
            base_f1 = best_f1
            for _ in range(3):
                improved = False
                for s in cur:
                    keep = cur[s]
                    cand_t, cand_f1 = keep, base_f1
                    for t in GRID:
                        cur[s] = float(t)
                        pred = np.zeros(len(yv), dtype=int)
                        for s2, t2 in cur.items():
                            sm = sv == s2
                            pred[sm] = (pv[sm] >= t2).astype(int)
                        f1 = f1_score(yv, pred, average="macro")
                        if f1 > cand_f1 + 1e-12:
                            cand_f1, cand_t = f1, float(t)
                    cur[s] = cand_t if cand_f1 >= base_f1 + min_gain else keep
                    if cur[s] != keep:
                        base_f1, improved = cand_f1, True
                if not improved:
                    break
        for s, t in cur.items():
            # A cell with essentially no positives in training (en-UK/google: 1 of 1817)
            # cannot support a fitted threshold, and a false positive there is pure loss
            # against a variety that has only a handful of true positives. Suppress it.
            if yv[sv == s].mean() < 0.01:
                t = 0.95
            thresholds[(str(var), str(s))] = t
    return thresholds


def rankify(x: np.ndarray) -> np.ndarray:
    """Map each column to uniform [0, 1] ranks within its own set.

    Used for ensembling: two backbones can rank examples equally well while emitting
    probabilities on quite different scales, and averaging the raw probabilities then
    lands the fitted threshold in the wrong place.
    """
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        r = x[:, c].argsort().argsort().astype(float)
        out[:, c] = r / max(len(r) - 1, 1)
    return out


def apply_thresholds(prob: np.ndarray, variety: np.ndarray, source: np.ndarray,
                     thresholds: dict[tuple[str, str], float], default: float = 0.5) -> np.ndarray:
    pred = np.zeros(len(prob), dtype=int)
    for i in range(len(prob)):
        key = (str(variety[i]), str(source[i]))
        t = thresholds.get(key)
        if t is None:  # unseen source at test time -> fall back to the variety mean
            same = [v for (vv, _), v in thresholds.items() if vv == str(variety[i])]
            t = float(np.mean(same)) if same else default
        pred[i] = int(prob[i] >= t)
    return pred


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probs(model, loader, device, amp_dtype) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            sent, sarc = model(**batch)
        p = np.stack(
            [
                F.softmax(sent.float(), dim=-1)[:, 1].cpu().numpy(),
                F.softmax(sarc.float(), dim=-1)[:, 1].cpu().numpy(),
            ],
            axis=1,
        )
        out.append(p)
    return np.concatenate(out, axis=0)


def train_one_fold(spec, tok, train_df, infer_loaders, device, args, seed, grad_ckpt=False,
                   bs_override=None, init_ckpt=None):
    if grad_ckpt:
        spec = ModelSpec(**{**spec.__dict__, "grad_ckpt": True})
    model = MultiTaskClassifier(spec.hf_name, n_msd=args.msd, p_drop=args.dropout, spec=spec)
    if spec.load_4bit:
        # bitsandbytes already places 4-bit weights on the GPU, and calling .to() on them
        # raises; only the fp32 heads still need moving.
        model.sent_head.to(device)
        model.sarc_head.to(device)
    else:
        model = model.to(device)
    if init_ckpt is not None:
        missing, unexpected = model.encoder.load_state_dict(
            torch.load(init_ckpt, map_location=device), strict=False)
        if missing or unexpected:
            print(f"      [init] loaded intermediate weights "
                  f"({len(missing)} missing, {len(unexpected)} unexpected)")

    bs = bs_override or spec.batch_size
    accum = max(1, int(round(spec.accum * (spec.batch_size / bs))))
    train_ds = TextDataset(train_df, tok, spec.max_len, True,
                           variety_aug=getattr(args, "variety_aug", 0.0))
    sampler = LengthGroupedSampler(train_ds.lengths, bs, seed=seed)
    loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, collate_fn=Collator(tok),
        num_workers=getattr(args, "num_workers", 0),
        persistent_workers=getattr(args, "num_workers", 0) > 0,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    steps = math.ceil(len(loader) / accum) * spec.epochs
    optimizer = build_optimizer(model, spec)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * spec.warmup_ratio), steps)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    # torch.amp.GradScaler is a >=2.4 API; this environment runs torch 2.1.2.
    _GradScaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    _use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    try:
        scaler = _GradScaler("cuda", enabled=_use_scaler)
    except TypeError:
        scaler = _GradScaler(enabled=_use_scaler)

    # Class-balanced weights: sarcasm is ~19% positive overall, ~0% for en-UK/google.
    w_sent = class_weights(train_df["sentiment"].to_numpy()).to(device)
    w_sarc = class_weights(train_df["sarcasm"].to_numpy()).to(device)

    model.train()
    t0 = time.time()
    for epoch in range(spec.epochs):
        sampler.set_epoch(epoch)
        running, nb = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            y_sent = batch.pop("y_sent").to(device)
            y_sarc = batch.pop("y_sarc").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                sent, sarc = model(**batch)
                if args.focal_gamma > 0:
                    # Focal is applied to sarcasm only: sentiment is already at 0.93-0.97 and
                    # has no hard-example problem to fix.
                    loss = (
                        F.cross_entropy(sent.float(), y_sent, weight=w_sent,
                                        label_smoothing=args.label_smoothing)
                        + args.sarcasm_weight
                        * focal_ce(sarc.float(), y_sarc, weight=w_sarc, gamma=args.focal_gamma,
                                   label_smoothing=args.label_smoothing)
                    )
                else:
                    loss = (
                        F.cross_entropy(sent.float(), y_sent, weight=w_sent, label_smoothing=args.label_smoothing)
                        + args.sarcasm_weight
                        * F.cross_entropy(sarc.float(), y_sarc, weight=w_sarc, label_smoothing=args.label_smoothing)
                    )
            scaler.scale(loss / accum).backward()
            running += loss.item()
            nb += 1
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"      epoch {epoch + 1}/{spec.epochs}  loss={running / max(nb, 1):.4f}  "
              f"({time.time() - t0:.0f}s, peak {peak:.1f} GiB)")

    preds = [predict_probs(model, dl, device, amp_dtype) for dl in infer_loaders]
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def pretrain_encoder(spec: ModelSpec, tok, device, args, key: str) -> Path:
    """Intermediate stage: fine-tune the encoder on external sarcasm data, once.

    Run once per backbone and cached, so all K folds start from the same adapted weights
    instead of paying the cost K times.
    """
    ckpt = args.artifacts / f"pretrained_{key}_{args.pretrain_epochs}ep.pt"  # per-artifacts-dir
    if ckpt.exists() and not args.overwrite:
        print(f"  [cache] intermediate encoder {ckpt.name}")
        return ckpt

    ext = load_external_sarcasm(args.artifacts / "external_sarcasm.csv")
    if args.pretrain_n and len(ext) > args.pretrain_n:
        per = args.pretrain_n // 2
        ext = (ext.groupby("sarcasm", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per), random_state=0))
                  .sample(frac=1.0, random_state=0).reset_index(drop=True))
    pt_len = args.pretrain_max_len or spec.max_len
    print(f"  intermediate task: {len(ext)} examples, {ext.sarcasm.mean():.1%} sarcastic, "
          f"max_len={pt_len}")
    model = MultiTaskClassifier(spec.hf_name, n_msd=args.msd, p_drop=args.dropout, spec=spec).to(device)
    ds = ExternalDataset(ext, tok, pt_len)
    pt_bs = args.pretrain_bs or spec.batch_size
    sampler = LengthGroupedSampler(ds.lengths, pt_bs, seed=0)
    loader = DataLoader(ds, batch_size=pt_bs, sampler=sampler,
                        collate_fn=Collator(tok),
                        num_workers=getattr(args, "num_workers", 0),
                        persistent_workers=getattr(args, "num_workers", 0) > 0,
                        pin_memory=device.type == "cuda")

    steps = math.ceil(len(loader) / spec.accum) * args.pretrain_epochs
    opt = build_optimizer(model, spec)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.1), steps)
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    w = class_weights(ext.sarcasm.to_numpy()).to(device)

    model.train()
    t0 = time.time()
    for ep in range(args.pretrain_epochs):
        sampler.set_epoch(ep)
        run, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            y = batch.pop("y_sarc").to(device)
            batch.pop("y_sent")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                _, sarc = model(**batch)
                loss = F.cross_entropy(sarc.float(), y, weight=w)
            (loss / spec.accum).backward()
            run += loss.item()
            nb += 1
            if (step + 1) % spec.accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
        print(f"    intermediate epoch {ep + 1}/{args.pretrain_epochs} loss={run / max(nb, 1):.4f} "
              f"({time.time() - t0:.0f}s)")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), ckpt)
    del model, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt


def focal_ce(logits, target, weight=None, gamma=2.0, label_smoothing=0.0):
    """Class-weighted focal loss (Lin et al., ICCV 2017).

    Sarcasm is ~19% positive overall and the hard positives (sarcastic text whose surface
    form reads literal) are exactly the ones plain cross-entropy stops learning from once
    the easy majority is fit. Focal down-weights well-classified examples by (1-p_t)^gamma
    so gradient mass stays on the hard cases.
    """
    ce = F.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing,
                         reduction="none")
    logp = F.log_softmax(logits, dim=-1)
    pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
    return ((1.0 - pt) ** gamma * ce).mean()


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)


def run_model(key: str, seed: int, pool_df: pd.DataFrame, target_df: pd.DataFrame, args, device):
    """K-fold train on `pool_df`; return (oof_probs, target_probs)."""
    spec = MODEL_ZOO[key]
    if args.epochs:
        spec = ModelSpec(**{**spec.__dict__, "epochs": args.epochs})
    if args.max_len:
        spec = ModelSpec(**{**spec.__dict__, "max_len": args.max_len})

    pt = f"_pt{args.pretrain_epochs}" if args.pretrain else ""
    tag = f"{key}_s{seed}_f{args.folds}_e{spec.epochs}_l{spec.max_len}{pt}_{args.run_tag}"
    oof_path = args.artifacts / f"{tag}_oof.npy"
    tgt_path = args.artifacts / f"{tag}_target.npy"
    if oof_path.exists() and tgt_path.exists() and not args.overwrite:
        print(f"  [cache] {tag}")
        return np.load(oof_path), np.load(tgt_path)

    tok = AutoTokenizer.from_pretrained(spec.hf_name)
    if tok.pad_token is None:  # causal LMs typically ship without one
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # last-token pooling indexes (n_tokens - 1)
    strat = (
        pool_df["variety"].astype(str) + "|" + pool_df["source"].astype(str) + "|"
        + pool_df["sentiment"].astype(str) + "|" + pool_df["sarcasm"].astype(str)
    ).to_numpy()
    # Merge strata that are too small for K-fold (e.g. en-UK/google sarcastic: n=1).
    vals, counts = np.unique(strat, return_counts=True)
    rare = set(vals[counts < args.folds])
    strat = np.array(["|".join(s.split("|")[:2]) if s in rare else s for s in strat])

    target_loader = DataLoader(
        TextDataset(target_df, tok, spec.max_len, False),
        batch_size=min(spec.batch_size * 2, 16), shuffle=False, collate_fn=Collator(tok), num_workers=0,
    )
    init_ckpt = pretrain_encoder(spec, tok, device, args, key) if args.pretrain else None

    oof = np.zeros((len(pool_df), 2), dtype=np.float64)
    tgt = np.zeros((len(target_df), 2), dtype=np.float64)

    extra_df = None
    if args.extra_train is not None:
        extra_df = load_csv(args.extra_train).dropna(subset=["sentiment", "sarcasm"]).reset_index(drop=True)
        for _t in TASKS:
            extra_df[_t] = extra_df[_t].astype(int)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(pool_df)), strat)):
        print(f"    fold {fold + 1}/{args.folds}  (train={len(tr_idx)}, val={len(va_idx)})")
        set_seed(seed * 1000 + fold)
        tr_df = pool_df.iloc[tr_idx].reset_index(drop=True)
        va_df = pool_df.iloc[va_idx].reset_index(drop=True)
        if extra_df is not None:
            tr_df = pd.concat([tr_df, extra_df], ignore_index=True)
            if fold == 0:
                print(f"      [extra] +{len(extra_df)} rows "
                      f"({int(extra_df.sarcasm.sum())} sarcastic) into the train half only")
        val_loader = DataLoader(
            TextDataset(va_df, tok, spec.max_len, False),
            batch_size=min(spec.batch_size * 2, 16), shuffle=False, collate_fn=Collator(tok), num_workers=0,
        )
        try:
            p_val, p_tgt = train_one_fold(spec, tok, tr_df, [val_loader, target_loader], device, args, seed,
                                          bs_override=args.bs, init_ckpt=init_ckpt)
        except torch.cuda.OutOfMemoryError:
            print("      [OOM] retrying with gradient checkpointing and half batch size")
            gc.collect()
            torch.cuda.empty_cache()
            p_val, p_tgt = train_one_fold(
                spec, tok, tr_df, [val_loader, target_loader], device, args, seed,
                grad_ckpt=True, bs_override=max(1, spec.batch_size // 2), init_ckpt=init_ckpt,
            )
        oof[va_idx] = p_val
        tgt += p_tgt / args.folds
        s, f = official_score(va_df, (p_val[:, 0] >= 0.5).astype(int), (p_val[:, 1] >= 0.5).astype(int))
        print(f"      fold@0.5 -> {', '.join(f'{k}={v:.4f}' for k, v in s.items())} | score={f:.4f}")
        if args.max_folds and fold + 1 >= args.max_folds:
            print(f"  [--max_folds {args.max_folds}] stopping early; not caching partial artifacts")
            return None, None

    args.artifacts.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)
    np.save(tgt_path, tgt)
    return oof, tgt


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"source", "variety", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="ALTA 2026 sentiment + sarcasm system")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--test", type=Path, default=None,
                    help="Unlabelled test CSV. If omitted, valid.csv is used as the target (dev phase).")
    ap.add_argument("--out", type=Path, default=Path("answer.csv"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    # roberta-large measured clearly weaker on this task (target @oracle 0.8326 vs
    # deberta 0.8712) and blending it in *lowered* the target score, so it is not a default.
    ap.add_argument("--models", type=str, default="deberta",
                    help=f"comma-separated, from: {', '.join(MODEL_ZOO)}")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                    help="train only the first N folds of the same K-fold split (cheap A/B test). "
                         "Artifacts are not cached and no blend is produced.")
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=None, help="override the per-model default")
    ap.add_argument("--max_len", type=int, default=None, help="override the per-model default")
    ap.add_argument("--bs", type=int, default=None, help="override batch size (accum is rescaled to match)")
    ap.add_argument("--pretrain", action="store_true",
                    help="intermediate-task fine-tuning on public sarcasm/irony corpora before "
                         "training on BESSTIE (encoder backbones only)")
    ap.add_argument("--pretrain_epochs", type=int, default=2)
    ap.add_argument("--pretrain_max_len", type=int, default=None,
                    help="shorter budget for the intermediate stage (SARC median is 10 words)")
    ap.add_argument("--pretrain_bs", type=int, default=None,
                    help="batch size for the intermediate stage only. At max_len 64 a bs of 8 "
                         "leaves the GPU ~11 percent utilised: the step is too small to hide "
                         "the dataloader. 64 keeps it saturated.")
    ap.add_argument("--pretrain_n", type=int, default=None,
                    help="subsample the external corpus to N rows (class-balanced)")
    ap.add_argument("--extra_train", type=Path, default=None,
                    help="extra labelled CSV appended to the TRAIN half of every fold only. "
                         "Never enters the validation half, so OOF stays honest. Used for the "
                         "BESSTIE en-IN subset: +272 sarcastic rows, and en-IN is not a "
                         "test variety so its labels are not test labels.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="dataloader workers; 0 leaves the GPU waiting on one CPU core")
    ap.add_argument("--variety_aug", type=float, default=0.0,
                    help="probability of neutralising the variety token on a training row")
    ap.add_argument("--sarcasm_weight", type=float, default=1.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="focal-loss gamma on the sarcasm head (0 = plain cross-entropy)")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--msd", type=int, default=5, help="multi-sample dropout passes")
    ap.add_argument("--no_per_source_threshold", action="store_true")
    ap.add_argument("--no_threshold_tuning", action="store_true")
    ap.add_argument("--train_on_valid", action="store_true",
                    help="add valid.csv to the training pool (use for the final test-phase run)")
    ap.add_argument("--blend", choices=["rank", "prob"], default="rank",
                    help="how to combine multiple runs: rank-average (calibration-invariant, "
                         "default) or probability-average. Ignored for a single run.")
    ap.add_argument("--blend_only", action="store_true", help="reuse cached artifacts, do not train")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--run_tag", type=str, default="v1")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (1 fold, 1 epoch, distilbert)")
    args = ap.parse_args()

    if args.quick:
        args.models, args.folds, args.epochs, args.max_len, args.run_tag = "distilbert", 2, 1, 128, "quick"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    train_df = load_csv(args.data_dir / "train.csv")
    valid_df = load_csv(args.data_dir / "valid.csv")

    dev_phase = args.test is None
    if dev_phase:
        target_df = valid_df.copy()
        pool_df = train_df.copy()
        if args.train_on_valid:
            print("[warn] --train_on_valid with no --test leaks the target into training; ignoring it.")
    else:
        target_df = load_csv(args.test)
        pool_df = pd.concat([train_df, valid_df], ignore_index=True) if args.train_on_valid else train_df.copy()
    pool_df = pool_df.dropna(subset=["sentiment", "sarcasm"]).reset_index(drop=True)
    for t in TASKS:
        pool_df[t] = pool_df[t].astype(int)
    print(f"pool={len(pool_df)}  target={len(target_df)}  phase={'dev (valid.csv)' if dev_phase else 'test'}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    unknown = [m for m in models if m not in MODEL_ZOO]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; available: {sorted(MODEL_ZOO)}")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    oofs, tgts, names = [], [], []
    for key in models:
        for seed in seeds:
            print(f"\n=== {key} (seed {seed}) ===")
            if args.blend_only:
                spec = MODEL_ZOO[key]
                ep = args.epochs or spec.epochs
                ml = args.max_len or spec.max_len
                tag = f"{key}_s{seed}_f{args.folds}_e{ep}_l{ml}_{args.run_tag}"
                o, t = args.artifacts / f"{tag}_oof.npy", args.artifacts / f"{tag}_target.npy"
                if not (o.exists() and t.exists()):
                    print(f"  [skip] no cached artifacts for {tag}")
                    continue
                oof, tgt = np.load(o), np.load(t)
            else:
                oof, tgt = run_model(key, seed, pool_df, target_df, args, device)
                if oof is None:  # --max_folds early stop
                    continue
            oofs.append(oof)
            tgts.append(tgt)
            names.append(f"{key}:{seed}")

    if not oofs:
        if args.max_folds:
            print("\n[--max_folds] A/B test complete; compare the fold@0.5 lines above.")
            return
        raise SystemExit("nothing to blend -- train at least one model first")

    # --- blend --------------------------------------------------------------
    # Averaging raw probabilities across architectures mixes differently-calibrated
    # scales, which throws off the OOF-fitted thresholds: on deberta+qwen4b the
    # probability blend scored 0.8356 on target while rank-averaging scored 0.8670
    # (both from an oracle ceiling of ~0.883). Ranks are calibration-invariant.
    # Seeds of one backbone are averaged together first, then architectures are combined with
    # equal weight. A flat average over runs would let whichever backbone has more seeds
    # dominate, which is what made deberta(x2)+qwen (0.8521) worse than deberta+qwen (0.8670):
    # the diversity between architectures is the thing carrying the ensemble.
    if len(oofs) > 1 and args.blend == "rank":
        arch_o, arch_t = {}, {}
        for nm, o, t in zip(names, oofs, tgts):
            key = nm.split(":")[0]
            arch_o.setdefault(key, []).append(rankify(o))
            arch_t.setdefault(key, []).append(rankify(t))
        oof = np.mean([np.mean(v, axis=0) for v in arch_o.values()], axis=0)
        tgt = np.mean([np.mean(v, axis=0) for v in arch_t.values()], axis=0)
        if len(arch_o) > 1:
            print(f"  architecture-balanced over {len(arch_o)} backbone(s): "
                  + ", ".join(f"{k}x{len(v)}" for k, v in arch_o.items()))
    else:
        oof = np.mean(oofs, axis=0)
        tgt = np.mean(tgts, axis=0)
    print(f"\n=== blend of {len(names)} run(s): {', '.join(names)} "
          f"[{args.blend if len(names) > 1 else 'single'}] ===")

    var_pool = pool_df["variety"].to_numpy()
    src_pool = pool_df["source"].to_numpy()
    var_tgt = target_df["variety"].to_numpy()
    src_tgt = target_df["source"].to_numpy()

    _, base = official_score(pool_df, (oof[:, 0] >= 0.5).astype(int), (oof[:, 1] >= 0.5).astype(int))
    print(f"OOF @0.5              score={base:.4f}")

    thresholds = {}
    if args.no_threshold_tuning:
        pred_sent_oof = (oof[:, 0] >= 0.5).astype(int)
        pred_sarc_oof = (oof[:, 1] >= 0.5).astype(int)
        pred_sent = (tgt[:, 0] >= 0.5).astype(int)
        pred_sarc = (tgt[:, 1] >= 0.5).astype(int)
    else:
        per_src = not args.no_per_source_threshold
        th_sent = tune_thresholds(oof[:, 0], pool_df["sentiment"].to_numpy(), var_pool, src_pool, per_src)
        th_sarc = tune_thresholds(oof[:, 1], pool_df["sarcasm"].to_numpy(), var_pool, src_pool, per_src)
        thresholds = {"sentiment": {f"{k[0]}/{k[1]}": v for k, v in th_sent.items()},
                      "sarcasm": {f"{k[0]}/{k[1]}": v for k, v in th_sarc.items()}}
        print("thresholds:", json.dumps(thresholds, indent=2))
        pred_sent_oof = apply_thresholds(oof[:, 0], var_pool, src_pool, th_sent)
        pred_sarc_oof = apply_thresholds(oof[:, 1], var_pool, src_pool, th_sarc)
        pred_sent = apply_thresholds(tgt[:, 0], var_tgt, src_tgt, th_sent)
        pred_sarc = apply_thresholds(tgt[:, 1], var_tgt, src_tgt, th_sarc)

    s, f = official_score(pool_df, pred_sent_oof, pred_sarc_oof)
    print("OOF tuned            ", ", ".join(f"{k}={v:.4f}" for k, v in s.items()), f"| score={f:.4f}")

    # --- write submission ---------------------------------------------------
    answer = target_df.copy()
    answer["sentiment"] = pred_sent
    answer["sarcasm"] = pred_sarc
    answer = answer[["source", "variety", "text", "sentiment", "sarcasm"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(args.out, index=False)
    zip_path = args.out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out, arcname="answer.csv")
    # Any full run overwrites answer.csv, so a later single-model run can silently replace a
    # better blend. This sidecar records what actually produced the file sitting on disk --
    # check it before submitting.
    meta = {"runs": names, "blend": args.blend if len(names) > 1 else "single",
            "folds": args.folds, "run_tag": args.run_tag,
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"\nwrote {args.out} and {zip_path}  ({len(answer)} rows)")
    print("predicted positives: sentiment=%.3f sarcasm=%.3f" % (pred_sent.mean(), pred_sarc.mean()))

    # --- honest score on the target when labels are available ---------------
    if all(c in target_df.columns for c in TASKS) and target_df[TASKS].notna().all().all():
        gold = target_df.copy()
        for t in TASKS:
            gold[t] = gold[t].astype(int)
        s, f = official_score(gold, pred_sent, pred_sarc)
        print("\n=== TARGET (labelled) ===")
        for k, v in s.items():
            print(f"f1-{k}: {v:.4f}")
        print(f"FINAL SCORE: {f:.4f}   (leaderboard #1 = 0.8754)")
        summary = {"target_scores": s, "target_final": f, "oof_final": float(base),
                   "runs": names, "thresholds": thresholds}
        (args.artifacts / f"summary_{args.run_tag}.json").write_text(json.dumps(summary, indent=2))
        meta["target_final"] = round(f, 4)
        meta["target_scores"] = {k: round(v, 4) for k, v in s.items()}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"provenance -> {args.out.with_suffix('.meta.json')}: {meta.get('target_final', 'n/a')} "
          f"from {', '.join(names)}")


if __name__ == "__main__":
    main()
