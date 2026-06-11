# -*- coding: utf-8 -*-
"""
run_normaware_proxy_pauc_cmtc.py

Dual-tail / feature-mode CMTC experiment script.

Main purpose:
1) Compare STFT, Log-mel, and MHP features.
2) Compare target-only Tail vs dual-stage Tail:
      source stage: CE or CE + Source Tail
      target stage: CosMetricHead with CE or CE + Target Tail, plus optional TC.
3) Produce all_metrics.csv, summary_mean_std.csv, delta_by_k.csv, overall_verdict.csv.

Label convention:
    0 = other/background
    1 = target

Expected data layout, robustly supported:
    data_targets/<domain>/other/**/*.wav
    data_targets/<domain>/target/**/*.wav

Example:
python run_dt_feature_cmtc.py --data_root data_targets \
  --tasks open+speedboat:uuv,open+uuv:speedboat \
  --seeds 2026 --k_shots 5,10,20 \
  --epochs_ce 10 --repeats 5 --query_cap_per_class 500 \
  --batch_size 16 --cuda \
  --feature_mode logmel --cache_dir cache_logmel_cmtc \
  --out_dir outputs_logmel_targettail_quick

Dual-tail example:
python run_dt_feature_cmtc.py --data_root data_targets \
  --tasks open+speedboat:uuv,open+uuv:speedboat \
  --seeds 2026 --k_shots 5,10,20 \
  --epochs_ce 10 --repeats 5 --query_cap_per_class 500 \
  --batch_size 16 --cuda \
  --feature_mode logmel --cache_dir cache_logmel_cmtc \
  --src_tail_weight 0.05 --src_tail_warmup_epochs 3 \
  --out_dir outputs_logmel_dualtail005_quick
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. General utilities
# ============================================================

AUDIO_EXTS = {".wav", ".wave"}
METRICS = [
    "acc", "f1", "auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01",
    "pd_tc", "pfa_tc", "threshold_tc",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_tasks(s: str) -> List[Tuple[List[str], str]]:
    tasks = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Task must be like open+speedboat:uuv, got {part!r}")
        src, tgt = part.split(":", 1)
        src_domains = [x.strip() for x in src.split("+") if x.strip()]
        tgt = tgt.strip()
        if not src_domains or not tgt:
            raise ValueError(f"Invalid task: {part!r}")
        tasks.append((src_domains, tgt))
    if not tasks:
        raise ValueError("No valid tasks.")
    return tasks


def task_name(src_domains: Sequence[str], target: str) -> str:
    return "+".join(src_domains) + "->" + target


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)


# ============================================================
# 2. Feature extraction: STFT / Log-mel / MHP
# ============================================================

def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 10.0,
    fmax: float = 800.0,
) -> np.ndarray:
    fmax = min(float(fmax), sr / 2.0)
    fmin = max(0.0, float(fmin))
    if fmax <= fmin:
        raise ValueError(f"Invalid mel range: fmin={fmin}, fmax={fmax}, sr={sr}")

    mel_min = hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])

        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)

        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[m - 1, center:right + 1] = (right - np.arange(center, right + 1)) / max(right - center, 1)

    empty = np.where(fb.sum(axis=1) <= 0)[0]
    for idx in empty:
        k = min(max(int(round((idx + 1) * (n_fft // 2) / (n_mels + 1))), 0), n_fft // 2)
        fb[idx, k] = 1.0
    return fb


def fix_time_bins(feat: np.ndarray, out_t: int = 96) -> np.ndarray:
    if feat.ndim != 2:
        raise ValueError(f"Expected [F,T], got {feat.shape}")
    f, t = feat.shape
    if t == out_t:
        return feat
    if t > out_t:
        start = (t - out_t) // 2
        return feat[:, start:start + out_t]
    pad_left = (out_t - t) // 2
    pad_right = out_t - t - pad_left
    return np.pad(feat, ((0, 0), (pad_left, pad_right)), mode="edge")


def normalize_feature(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    feat = feat.astype(np.float32, copy=False)
    if feat.ndim == 2:
        return (feat - float(feat.mean())) / (float(feat.std()) + eps)
    if feat.ndim == 3:
        out = np.empty_like(feat, dtype=np.float32)
        for c in range(feat.shape[0]):
            out[c] = (feat[c] - float(feat[c].mean())) / (float(feat[c].std()) + eps)
        return out
    raise ValueError(f"Expected [F,T] or [C,F,T], got {feat.shape}")


def power_to_db_np(power: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """librosa.power_to_db(..., top_db=None) compatible enough for this experiment."""
    power = np.asarray(power, dtype=np.float32)
    return (10.0 * np.log10(np.maximum(power, amin)) - 10.0 * np.log10(max(ref, amin))).astype(np.float32)


def maybe_normalize_feature(feat: np.ndarray, norm_mode: str) -> np.ndarray:
    """sample: per-sample norm; none/global: keep raw here. global is applied after source loading."""
    if norm_mode == "sample":
        return normalize_feature(feat)
    if norm_mode in {"none", "global"}:
        return feat.astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature_norm={norm_mode}")


def stft_power(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    window: str = "hamming",
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = wav - float(wav.mean())
    wav = wav / (float(np.max(np.abs(wav))) + 1e-8)

    noverlap = int(win_length - hop_length)
    _, _, zxx = signal.stft(
        wav,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return (np.abs(zxx).astype(np.float32) ** 2)


def hpss_power(
    power: np.ndarray,
    harmonic_kernel: int = 17,
    percussive_kernel: int = 17,
    power_exponent: float = 2.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    harmonic_kernel = int(max(3, harmonic_kernel))
    percussive_kernel = int(max(3, percussive_kernel))
    if harmonic_kernel % 2 == 0:
        harmonic_kernel += 1
    if percussive_kernel % 2 == 0:
        percussive_kernel += 1

    # Harmonic component: horizontal continuity over time.
    h_smooth = median_filter(power, size=(1, harmonic_kernel), mode="nearest")
    # Percussive component: vertical continuity over frequency.
    p_smooth = median_filter(power, size=(percussive_kernel, 1), mode="nearest")

    h = np.power(h_smooth + eps, power_exponent)
    p = np.power(p_smooth + eps, power_exponent)
    denom = h + p + eps
    mask_h = h / denom
    mask_p = p / denom
    return power * mask_h.astype(np.float32), power * mask_p.astype(np.float32)


def wav_to_stft_feature(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    freq_bins: int = 128,
    out_t: int = 96,
    norm_mode: str = "sample",
    log_power_mode: str = "log1p",
    window: str = "hamming",
) -> np.ndarray:
    power = stft_power(wav, sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    feat = power_to_db_np(power) if log_power_mode == "db" else np.log1p(power)
    if feat.shape[0] >= freq_bins:
        feat = feat[:freq_bins, :]
    else:
        feat = np.pad(feat, ((0, freq_bins - feat.shape[0]), (0, 0)), mode="edge")
    feat = fix_time_bins(feat, out_t=out_t)
    feat = maybe_normalize_feature(feat, norm_mode)
    return feat[None, :, :].astype(np.float32)


def wav_to_logmel_or_mhp_feature(
    wav: np.ndarray,
    sr: int,
    feature_mode: str = "logmel",
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    n_mels: int = 128,
    out_t: int = 96,
    fmin: float = 10.0,
    fmax: float = 800.0,
    hpss_harmonic_kernel: int = 17,
    hpss_percussive_kernel: int = 17,
    norm_mode: str = "sample",
    log_power_mode: str = "log1p",
    window: str = "hamming",
) -> np.ndarray:
    if feature_mode not in {"logmel", "mhp"}:
        raise ValueError(f"feature_mode must be logmel or mhp, got {feature_mode}")

    power = stft_power(wav=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    fb = build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    mel_power = fb @ power
    logmel = power_to_db_np(mel_power) if log_power_mode == "db" else np.log1p(mel_power)
    logmel = fix_time_bins(logmel, out_t=out_t)

    if feature_mode == "logmel":
        return maybe_normalize_feature(logmel, norm_mode)[None, :, :].astype(np.float32)

    h_power, p_power = hpss_power(
        power,
        harmonic_kernel=hpss_harmonic_kernel,
        percussive_kernel=hpss_percussive_kernel,
    )
    h_mel_power = fb @ h_power
    p_mel_power = fb @ p_power
    hmel = fix_time_bins(power_to_db_np(h_mel_power) if log_power_mode == "db" else np.log1p(h_mel_power), out_t=out_t)
    pmel = fix_time_bins(power_to_db_np(p_mel_power) if log_power_mode == "db" else np.log1p(p_mel_power), out_t=out_t)
    mhp = np.stack([logmel, hmel, pmel], axis=0)
    return maybe_normalize_feature(mhp, norm_mode).astype(np.float32)


def infer_in_channels(feature_mode: str, single_to_rgb: bool = False) -> int:
    if feature_mode == "mhp":
        return 3
    return 3 if single_to_rgb else 1


# ============================================================
# 3. Data loading and caching
# ============================================================

def read_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        maxv = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(maxv)
    else:
        data = data.astype(np.float32)

    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        up = target_sr // g
        down = sr // g
        data = signal.resample_poly(data, up, down).astype(np.float32)
    return data.reshape(-1)


def segment_wave(wav: np.ndarray, sr: int, segment_sec: float, hop_sec: float) -> List[np.ndarray]:
    seg_len = max(1, int(round(segment_sec * sr)))
    hop_len = max(1, int(round(hop_sec * sr)))
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    if wav.shape[0] <= seg_len:
        pad = seg_len - wav.shape[0]
        if pad > 0:
            wav = np.pad(wav, (0, pad), mode="constant")
        return [wav[:seg_len]]

    out = []
    for start in range(0, wav.shape[0] - seg_len + 1, hop_len):
        out.append(wav[start:start + seg_len])
    if not out:
        out.append(wav[:seg_len])
    return out


def find_label_files(domain_dir: Path) -> Dict[int, List[Path]]:
    """Return {0: other files, 1: target files}."""
    label_files: Dict[int, List[Path]] = {0: [], 1: []}
    label_aliases = {
        0: ["other", "background", "bg", "negative", "neg", "0"],
        1: ["target", "positive", "pos", "1"],
    }

    for label, aliases in label_aliases.items():
        for alias in aliases:
            d = domain_dir / alias
            if d.exists() and d.is_dir():
                label_files[label].extend([p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXTS])

    # Fallback: infer from path parts / file names.
    if not label_files[0] or not label_files[1]:
        all_files = [p for p in domain_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        for p in all_files:
            text = "/".join([x.lower() for x in p.parts[-4:]])
            if any(tok in text for tok in ["other", "background", "negative", "neg"]):
                if p not in label_files[0]:
                    label_files[0].append(p)
            elif any(tok in text for tok in ["target", "positive", "pos"]):
                if p not in label_files[1]:
                    label_files[1].append(p)

    label_files[0] = sorted(set(label_files[0]))
    label_files[1] = sorted(set(label_files[1]))
    return label_files


def feature_cache_name(domain: str, args: argparse.Namespace) -> str:
    parts = [
        f"domain={domain}",
        f"mode={args.feature_mode}",
        f"sr={args.target_sr}",
        f"seg={args.segment_sec}",
        f"hop={args.segment_hop_sec}",
        f"nfft={args.n_fft}",
        f"hoplen={args.hop_length}",
        f"win={args.win_length}",
        f"t={args.feature_time_bins}",
        f"stftF={args.stft_freq_bins}",
        f"mel={args.mel_n_mels}",
        f"fmin={args.mel_fmin}",
        f"fmax={args.mel_fmax}",
        f"hpss={args.hpss_harmonic_kernel}-{args.hpss_percussive_kernel}",
        f"norm={args.feature_norm}",
        f"logpow={args.log_power_mode}",
        f"window={args.stft_window}",
    ]
    tag = stable_hash("|".join(parts), n=10)
    return f"{domain}_{args.feature_mode}_{tag}.npz"


def extract_feature_from_segment(seg: np.ndarray, sr: int, args: argparse.Namespace) -> np.ndarray:
    if args.feature_mode == "stft":
        return wav_to_stft_feature(
            seg,
            sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            freq_bins=args.stft_freq_bins,
            out_t=args.feature_time_bins,
            norm_mode=args.feature_norm,
            log_power_mode=args.log_power_mode,
            window=args.stft_window,
        )
    if args.feature_mode in {"logmel", "mhp"}:
        return wav_to_logmel_or_mhp_feature(
            seg,
            sr,
            feature_mode=args.feature_mode,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            n_mels=args.mel_n_mels,
            out_t=args.feature_time_bins,
            fmin=args.mel_fmin,
            fmax=args.mel_fmax,
            hpss_harmonic_kernel=args.hpss_harmonic_kernel,
            hpss_percussive_kernel=args.hpss_percussive_kernel,
            norm_mode=args.feature_norm,
            log_power_mode=args.log_power_mode,
            window=args.stft_window,
        )
    raise ValueError(f"Unknown feature_mode={args.feature_mode}")


def load_domain_features(data_root: Path, domain: str, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    ensure_dir(Path(args.cache_dir))
    cache_path = Path(args.cache_dir) / feature_cache_name(domain, args)
    if cache_path.exists() and not args.rebuild_cache:
        d = np.load(cache_path, allow_pickle=False)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)

    domain_dir = data_root / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    label_files = find_label_files(domain_dir)
    if not label_files[0] or not label_files[1]:
        raise RuntimeError(
            f"Could not find both classes under {domain_dir}. "
            f"Found other={len(label_files[0])}, target={len(label_files[1])}."
        )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    print(f"[DATA] extracting domain={domain}: other files={len(label_files[0])}, target files={len(label_files[1])}")

    for label in [0, 1]:
        for p in label_files[label]:
            try:
                wav = read_wav_mono(p, target_sr=args.target_sr)
                segs = segment_wave(wav, args.target_sr, args.segment_sec, args.segment_hop_sec)
                for seg in segs:
                    feat = extract_feature_from_segment(seg, args.target_sr, args)
                    X_list.append(feat)
                    y_list.append(label)
            except Exception as e:
                print(f"[WARN] failed to process {p}: {e}")

    if not X_list:
        raise RuntimeError(f"No features extracted for domain={domain}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"[CACHE] saved {cache_path} X={X.shape} y={np.bincount(y, minlength=2).tolist()}")
    return X, y


def apply_global_feature_norm(X_src: np.ndarray, X_tgt: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize with source-domain statistics only.

    This is closer to the uploaded paper code's dataset-level normalization,
    but avoids using target-query statistics.
    """
    mean = X_src.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = X_src.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ((X_src - mean) / std).astype(np.float32), ((X_tgt - mean) / std).astype(np.float32)


def maybe_replicate_single_to_rgb(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.single_to_rgb and args.feature_mode != "mhp" and X.shape[1] == 1:
        return np.repeat(X, 3, axis=1).astype(np.float32)
    return X.astype(np.float32, copy=False)


# ============================================================
# 4. Model
# ============================================================

class SmallTFEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, emb_dim: int = 128, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        z = self.proj(h)
        return z


class ResNet18Encoder(nn.Module):
    """Optional paper-style ResNet18 encoder.

    The uploaded paper code used torchvision.models.resnet18(pretrained=False),
    removed the original fc layer, then projected to a 128-d embedding.
    We keep the import lazy so the script still runs without torchvision unless
    --backbone resnet18 is requested.
    """
    def __init__(self, in_channels: int = 3, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        import torchvision
        try:
            net = torchvision.models.resnet18(weights=None)
        except TypeError:
            net = torchvision.models.resnet18(pretrained=False)
        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
        last_size = net.fc.in_features
        net.fc = nn.Identity()
        self.trunk = net
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_size, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(x))


class CosMetricHead(nn.Module):
    def __init__(self, emb_dim: int, scale: float = 10.0, init_anchors: Optional[torch.Tensor] = None):
        super().__init__()
        if init_anchors is None:
            init = torch.randn(2, emb_dim) * 0.02
        else:
            init = init_anchors.detach().float().clone()
        self.anchors = nn.Parameter(init)
        self.scale = float(scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zn = F.normalize(z, dim=1)
        an = F.normalize(self.anchors, dim=1)
        return self.scale * (zn @ an.t())


class CalibratedProxyCosMetricHead(nn.Module):
    """Calibrated proxy-cosine metric head.

    This is still a metric/proxy head: logits are cosine similarities
    between normalized embeddings and normalized class proxies. Compared
    with plain CosMetricHead, it adds a learnable scale and class bias.
    The bias is important in two-class low-false-alarm detection because
    it gives the metric score a learnable offset similar to LinearHead,
    while keeping the angular proxy structure.
    """
    def __init__(
        self,
        emb_dim: int,
        init_scale: float = 10.0,
        init_anchors: Optional[torch.Tensor] = None,
        learn_scale: bool = True,
    ):
        super().__init__()
        if init_anchors is None:
            init = torch.randn(2, emb_dim) * 0.02
        else:
            init = init_anchors.detach().float().clone()
        self.anchors = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(2))
        init_scale = max(float(init_scale), 1e-3)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.register_buffer("log_scale", torch.log(torch.tensor(init_scale, dtype=torch.float32)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zn = F.normalize(z, dim=1)
        an = F.normalize(self.anchors, dim=1)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        return scale * (zn @ an.t()) + self.bias


class NormAwareCalibratedProxyCosMetricHead(nn.Module):
    """Norm-aware calibrated proxy-cosine metric head.

    Motivation:
      * Plain proxy cosine heads use only the direction of z and discard ||z||.
      * A LinearHead can implicitly use both direction and norm because w^T z = ||w|| ||z|| cos(w,z).
      * In weak-signal detection, embedding norm may encode quality/SNR/confidence.

    Logits:
        logit_c = scale * cos(normalize(z), normalize(a_c)) + bias_c + gamma_c * log(1 + ||z||_2)

    This keeps the class-proxy cosine metric structure while adding a small,
    learnable norm-aware calibration term.
    """
    def __init__(
        self,
        emb_dim: int,
        init_scale: float = 10.0,
        init_anchors: Optional[torch.Tensor] = None,
        learn_scale: bool = True,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        if init_anchors is None:
            init = torch.randn(2, emb_dim) * 0.02
        else:
            init = init_anchors.detach().float().clone()
        self.anchors = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(2))
        self.gamma = nn.Parameter(torch.full((2,), float(init_gamma)))
        init_scale = max(float(init_scale), 1e-3)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.register_buffer("log_scale", torch.log(torch.tensor(init_scale, dtype=torch.float32)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = torch.norm(z, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        zn = z / z_norm
        an = F.normalize(self.anchors, dim=1)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        r = torch.log1p(z_norm).squeeze(1)
        logits = scale * (zn @ an.t()) + self.bias
        logits = logits + r[:, None] * self.gamma[None, :]
        return logits


class RegularizedMahaProxyHead(nn.Module):
    """Regularized diagonal Mahalanobis proxy-distance head.

    This is a genuine proxy-distance metric head. For each class proxy p_c,

        d_c(z) = mean_j exp(log_q_j) * (z_j - p_{c,j})^2
        logit_c = -scale * d_c(z) + bias_c

    The target-vs-other detection score is:

        D(z) = logit_target - logit_other
             = scale * [d_other(z) - d_target(z)] + bias_delta

    Why this head is used here:
      * A LinearHead can implicitly use both direction and norm of z.
      * A plain cosine head discards norm and only keeps angle.
      * This Mahalanobis proxy head keeps a distance-metric interpretation,
        but it uses the raw embedding and therefore can absorb the useful
        magnitude/dimension-wise distance information that makes LinearHead strong.
      * Regularization anchors proxies near class means and keeps the diagonal
        metric near identity, which is important under K-shot support.
    """
    def __init__(
        self,
        emb_dim: int,
        init_scale: float = 10.0,
        init_proxies: Optional[torch.Tensor] = None,
        learn_scale: bool = True,
    ):
        super().__init__()
        if init_proxies is None:
            init = torch.randn(2, emb_dim) * 0.02
        else:
            init = init_proxies.detach().float().clone()
        self.proxies = nn.Parameter(init)
        self.register_buffer("init_proxies", init.clone())
        self.log_diag = nn.Parameter(torch.zeros(emb_dim))
        self.bias = nn.Parameter(torch.zeros(2))
        init_scale = max(float(init_scale), 1e-3)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.register_buffer("log_scale", torch.log(torch.tensor(init_scale, dtype=torch.float32)))

    def distances(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, D], proxies: [2, D] -> distances [N, 2]
        q = self.log_diag.exp().clamp(1e-3, 100.0)
        diff = z[:, None, :] - self.proxies[None, :, :]
        # mean over dimensions prevents the logit scale from depending too much on emb_dim.
        d = (diff.pow(2) * q[None, None, :]).mean(dim=2)
        return d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d = self.distances(z)
        scale = self.log_scale.exp().clamp(0.01, 100.0)
        logits = -scale * d + self.bias[None, :]
        return logits

    def regularization(self, proxy_l2: float = 1e-3, diag_l2: float = 1e-3, bias_l2: float = 0.0) -> torch.Tensor:
        reg = self.proxies.new_tensor(0.0)
        if proxy_l2 > 0:
            reg = reg + float(proxy_l2) * (self.proxies - self.init_proxies).pow(2).mean()
        if diag_l2 > 0:
            reg = reg + float(diag_l2) * self.log_diag.pow(2).mean()
        if bias_l2 > 0:
            reg = reg + float(bias_l2) * self.bias.pow(2).mean()
        return reg


class LinearEquivalentProxyDistanceHead(nn.Module):
    """Linear-equivalent squared-distance proxy head.

    In binary classification, a linear score D(z)=w^T z + b can be written exactly
    as a difference of squared Euclidean distances:

        D(z) = -||z - p_t||^2 + ||z - p_o||^2 + b_delta

    by choosing p_t=w/4 and p_o=-w/4. Therefore this head starts from the
    strongest target-domain LinearHead but represents the decision function as a
    genuine proxy-distance metric. Optional Tail fine-tuning can then improve
    low-false-alarm behavior while an anchor regularizer prevents overfitting.
    """
    def __init__(self, emb_dim: int, w_delta: torch.Tensor, b_delta: torch.Tensor):
        super().__init__()
        w_delta = w_delta.detach().float().reshape(-1)
        b_delta = b_delta.detach().float().reshape(())
        p_other = -0.25 * w_delta
        p_target = 0.25 * w_delta
        init_proxies = torch.stack([p_other, p_target], dim=0)
        init_bias = torch.stack([torch.tensor(0.0, dtype=torch.float32, device=w_delta.device), b_delta.to(w_delta.device)])
        self.proxies = nn.Parameter(init_proxies.clone())
        self.bias = nn.Parameter(init_bias.clone())
        self.register_buffer("init_proxies", init_proxies.clone())
        self.register_buffer("init_bias", init_bias.clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        diff = z[:, None, :] - self.proxies[None, :, :]
        d = diff.pow(2).sum(dim=2)
        logits = -d + self.bias[None, :]
        return logits

    def regularization(self, proxy_l2: float = 1e-3, bias_l2: float = 1e-4) -> torch.Tensor:
        reg = self.proxies.new_tensor(0.0)
        if proxy_l2 > 0:
            reg = reg + float(proxy_l2) * (self.proxies - self.init_proxies).pow(2).mean()
        if bias_l2 > 0:
            reg = reg + float(bias_l2) * (self.bias - self.init_bias).pow(2).mean()
        return reg


# ============================================================
# 5. Losses and metrics
# ============================================================

def tail_loss_from_detection_scores(
    det_score: torch.Tensor,
    y: torch.Tensor,
    q: float = 0.95,
    margin: float = 0.10,
) -> torch.Tensor:
    if det_score.ndim != 1:
        det_score = det_score.reshape(-1)
    y = y.reshape(-1).long()
    other_scores = det_score[y == 0]
    target_scores = det_score[y == 1]
    if other_scores.numel() < 2 or target_scores.numel() < 1:
        return det_score.new_tensor(0.0)
    other_tail = torch.quantile(other_scores, q)
    target_mean = target_scores.mean()
    return F.relu(other_tail - target_mean + margin)


def pauc_pairwise_loss(
    det_score: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.00,
    beta: float = 0.10,
    margin: float = 0.10,
) -> torch.Tensor:
    """Source-stage pAUC-style pairwise ranking loss.

    This follows the uploaded xuan_loss idea: sort negative scores descending,
    select a low-FPR negative interval [alpha, beta], and force positive scores
    to exceed those hard-negative scores by a margin.
    """
    if det_score.ndim != 1:
        det_score = det_score.reshape(-1)
    y = y.reshape(-1).long()
    pos = det_score[y == 1]
    neg = det_score[y == 0]
    if pos.numel() < 1 or neg.numel() < 2:
        return det_score.new_tensor(0.0)
    neg_sorted, _ = torch.sort(neg, descending=True)
    n = int(neg_sorted.numel())
    start = int(max(0.0, min(1.0, alpha)) * n)
    end = int(max(0.0, min(1.0, beta)) * n)
    if end <= start:
        end = start + 1
    end = min(end, n)
    hard_neg = neg_sorted[start:end]
    diff = pos.reshape(-1, 1) - hard_neg.reshape(1, -1)
    return torch.clamp(margin - diff, min=0.0).pow(2).mean()


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def pd_at_pfa(y_true: np.ndarray, score: np.ndarray, pfa: float = 0.05) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    neg = score[y_true == 0]
    pos = score[y_true == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - pfa))
    pd = float(np.mean(pos >= thr))
    return pd, thr


def metrics_from_scores(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.0,
    support_other_score: Optional[np.ndarray] = None,
    tc_quantile: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score > threshold).astype(int)

    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, pred)) if y_true.size else float("nan")
    out["f1"] = float(f1_score(y_true, pred, zero_division=0)) if y_true.size else float("nan")
    out["auc"] = safe_auc(y_true, score)
    out["pd_at_pfa_0.05"], _ = pd_at_pfa(y_true, score, pfa=0.05)
    out["pd_at_pfa_0.01"], _ = pd_at_pfa(y_true, score, pfa=0.01)

    if support_other_score is not None and len(support_other_score) > 0:
        threshold_tc = float(np.quantile(np.asarray(support_other_score).astype(float), tc_quantile))
        pred_tc = (score > threshold_tc).astype(int)
        pos = (y_true == 1)
        neg = (y_true == 0)
        out["threshold_tc"] = threshold_tc
        out["pd_tc"] = float(np.mean(pred_tc[pos] == 1)) if np.any(pos) else float("nan")
        out["pfa_tc"] = float(np.mean(pred_tc[neg] == 1)) if np.any(neg) else float("nan")
        # For TC methods we overwrite acc/f1 later in the row creation if needed.
        out["acc_tc"] = float(accuracy_score(y_true, pred_tc)) if y_true.size else float("nan")
        out["f1_tc"] = float(f1_score(y_true, pred_tc, zero_division=0)) if y_true.size else float("nan")
    else:
        out["threshold_tc"] = float("nan")
        out["pd_tc"] = float("nan")
        out["pfa_tc"] = float("nan")
        out["acc_tc"] = float("nan")
        out["f1_tc"] = float("nan")
    return out


# ============================================================
# 6. Training helpers
# ============================================================

def train_source_encoder(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[SmallTFEncoder, nn.Linear]:
    set_seed(seed)
    in_channels = infer_in_channels(args.feature_mode, args.single_to_rgb)
    if args.backbone == "small":
        encoder = SmallTFEncoder(in_channels=in_channels, emb_dim=args.emb_dim, width=args.encoder_width, dropout=args.dropout).to(device)
    elif args.backbone == "resnet18":
        encoder = ResNet18Encoder(in_channels=in_channels, emb_dim=args.emb_dim, dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Unknown backbone={args.backbone}")
    classifier = nn.Linear(args.emb_dim, 2).to(device)

    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr_ce,
        weight_decay=args.weight_decay,
    )

    for ep in range(int(args.epochs_ce)):
        encoder.train(); classifier.train()
        total = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            z = encoder(xb)
            logits = classifier(z)
            loss_ce = F.cross_entropy(logits, yb)
            loss = loss_ce

            det_score = logits[:, 1] - logits[:, 0]

            if args.src_tail_weight > 0 and ep >= args.src_tail_warmup_epochs:
                loss_src_tail = tail_loss_from_detection_scores(
                    det_score=det_score,
                    y=yb,
                    q=args.src_tail_quantile,
                    margin=args.src_tail_margin,
                )
                loss = loss + args.src_tail_weight * loss_src_tail

            if args.src_pauc_weight > 0 and ep >= args.src_pauc_warmup_epochs:
                loss_src_pauc = pauc_pairwise_loss(
                    det_score=det_score,
                    y=yb,
                    alpha=args.src_pauc_alpha,
                    beta=args.src_pauc_beta,
                    margin=args.src_pauc_margin,
                )
                loss = loss + args.src_pauc_weight * loss_src_pauc

            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            n += xb.shape[0]
        if args.verbose and (ep == 0 or ep == args.epochs_ce - 1 or (ep + 1) % 10 == 0):
            print(f"  [SRC] epoch {ep+1}/{args.epochs_ce} loss={total/max(n,1):.4f}")

    return encoder, classifier


@torch.no_grad()
def encode_array(encoder: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    encoder.eval()
    outs: List[np.ndarray] = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        z = encoder(xb)
        outs.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def class_means(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    means = []
    for c in [0, 1]:
        if torch.any(y == c):
            means.append(z[y == c].mean(dim=0))
        else:
            means.append(torch.zeros(z.shape[1], device=z.device, dtype=z.dtype))
    return torch.stack(means, dim=0)


def cosine_score_np(z: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    zt = torch.from_numpy(z).float()
    at = torch.from_numpy(anchors).float()
    zn = F.normalize(zt, dim=1)
    an = F.normalize(at, dim=1)
    logits = zn @ an.t()
    return (logits[:, 1] - logits[:, 0]).numpy()


def train_linear_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    head = nn.Linear(z_sup.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_linear, weight_decay=args.weight_decay)
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)
    for _ in range(int(args.linear_epochs)):
        opt.zero_grad(set_to_none=True)
        logits = head(zt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


def train_cos_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    use_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)
    init = class_means(zt, yt)
    head = CosMetricHead(emb_dim=z_sup.shape[1], scale=args.cos_scale, init_anchors=init).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_cos, weight_decay=args.weight_decay)

    for _ in range(int(args.cos_epochs)):
        opt.zero_grad(set_to_none=True)
        logits = head(zt)
        loss = F.cross_entropy(logits, yt)
        if use_tail and args.cos_tail_weight > 0:
            det_score = logits[:, 1] - logits[:, 0]
            loss_tail = tail_loss_from_detection_scores(
                det_score=det_score,
                y=yt,
                q=args.cos_tail_quantile,
                margin=args.cos_tail_margin,
            )
            loss = loss + args.cos_tail_weight * loss_tail
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


def train_calibrated_proxy_cos_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    use_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)
    init = class_means(zt, yt)
    head = CalibratedProxyCosMetricHead(
        emb_dim=z_sup.shape[1],
        init_scale=args.calib_cos_scale,
        init_anchors=init,
        learn_scale=(not args.calib_cos_fixed_scale),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_calib_cos, weight_decay=args.weight_decay)

    for _ in range(int(args.calib_cos_epochs)):
        opt.zero_grad(set_to_none=True)
        logits = head(zt)
        loss = F.cross_entropy(logits, yt)
        if use_tail and args.cos_tail_weight > 0:
            det_score = logits[:, 1] - logits[:, 0]
            loss_tail = tail_loss_from_detection_scores(
                det_score=det_score,
                y=yt,
                q=args.cos_tail_quantile,
                margin=args.cos_tail_margin,
            )
            loss = loss + args.cos_tail_weight * loss_tail
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


def train_normaware_calibrated_proxy_cos_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    use_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)
    init = class_means(zt, yt)
    head = NormAwareCalibratedProxyCosMetricHead(
        emb_dim=z_sup.shape[1],
        init_scale=args.na_calib_cos_scale,
        init_anchors=init,
        learn_scale=(not args.na_calib_cos_fixed_scale),
        init_gamma=args.na_calib_init_gamma,
    ).to(device)

    # Separate parameter groups: anchors/bias/scale learn normally; gamma is often
    # sensitive, so allow a smaller LR multiplier.
    gamma_params = [head.gamma]
    other_params = [p for n, p in head.named_parameters() if n != "gamma"]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr_na_calib_cos, "weight_decay": args.weight_decay},
            {"params": gamma_params, "lr": args.lr_na_calib_cos * args.na_calib_gamma_lr_mult, "weight_decay": 0.0},
        ]
    )

    for _ in range(int(args.na_calib_cos_epochs)):
        opt.zero_grad(set_to_none=True)
        logits = head(zt)
        loss = F.cross_entropy(logits, yt)
        if use_tail and args.cos_tail_weight > 0:
            det_score = logits[:, 1] - logits[:, 0]
            loss_tail = tail_loss_from_detection_scores(
                det_score=det_score,
                y=yt,
                q=args.cos_tail_quantile,
                margin=args.cos_tail_margin,
            )
            loss = loss + args.cos_tail_weight * loss_tail
        if args.na_calib_gamma_l2 > 0:
            loss = loss + args.na_calib_gamma_l2 * head.gamma.pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


def train_reg_maha_proxy_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    use_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)
    init = class_means(zt, yt)
    head = RegularizedMahaProxyHead(
        emb_dim=z_sup.shape[1],
        init_scale=args.maha_scale,
        init_proxies=init,
        learn_scale=(not args.maha_fixed_scale),
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_maha, weight_decay=args.weight_decay)

    for _ in range(int(args.maha_epochs)):
        opt.zero_grad(set_to_none=True)
        logits = head(zt)
        loss = F.cross_entropy(logits, yt)
        if use_tail and args.cos_tail_weight > 0:
            det_score = logits[:, 1] - logits[:, 0]
            loss_tail = tail_loss_from_detection_scores(
                det_score=det_score,
                y=yt,
                q=args.cos_tail_quantile,
                margin=args.cos_tail_margin,
            )
            loss = loss + args.cos_tail_weight * loss_tail
        loss = loss + head.regularization(
            proxy_l2=args.maha_proxy_l2,
            diag_l2=args.maha_diag_l2,
            bias_l2=args.maha_bias_l2,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


def train_lineq_proxy_head(
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    use_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train LinearHead, convert it exactly to a distance-proxy head, then optionally fine-tune with Tail.

    If use_tail=False and args.lineq_finetune_epochs=0, this is a distance-metric
    reparameterization of the target-domain linear head. If use_tail=True, it
    starts from that strong boundary and performs low-false-alarm tail adaptation
    in the proxy-distance parameterization.
    """
    zt = torch.from_numpy(z_sup).float().to(device)
    yt = torch.from_numpy(y_sup).long().to(device)

    # Step 1: fit target-domain linear head on support.
    lin = nn.Linear(z_sup.shape[1], 2).to(device)
    opt_lin = torch.optim.AdamW(lin.parameters(), lr=args.lr_linear, weight_decay=args.weight_decay)
    for _ in range(int(args.linear_epochs)):
        opt_lin.zero_grad(set_to_none=True)
        logits = lin(zt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt_lin.step()

    with torch.no_grad():
        w_delta = lin.weight[1] - lin.weight[0]
        b_delta = lin.bias[1] - lin.bias[0]

    # Step 2: convert linear score exactly to squared-distance proxy score.
    head = LinearEquivalentProxyDistanceHead(
        emb_dim=z_sup.shape[1],
        w_delta=w_delta,
        b_delta=b_delta,
    ).to(device)

    # Step 3: optionally fine-tune in metric parameterization.
    n_epochs = int(args.lineq_epochs if use_tail else args.lineq_finetune_epochs)
    if n_epochs > 0:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr_lineq, weight_decay=args.weight_decay)
        for _ in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            logits = head(zt)
            loss = F.cross_entropy(logits, yt)
            if use_tail and args.cos_tail_weight > 0:
                det_score = logits[:, 1] - logits[:, 0]
                loss_tail = tail_loss_from_detection_scores(
                    det_score=det_score,
                    y=yt,
                    q=args.cos_tail_quantile,
                    margin=args.cos_tail_margin,
                )
                loss = loss + args.cos_tail_weight * loss_tail
            loss = loss + head.regularization(
                proxy_l2=args.lineq_proxy_l2,
                bias_l2=args.lineq_bias_l2,
            )
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits_sup = head(zt)
        zq = torch.from_numpy(z_query).float().to(device)
        logits_q = head(zq)
    score_sup = (logits_sup[:, 1] - logits_sup[:, 0]).detach().cpu().numpy()
    score_q = (logits_q[:, 1] - logits_q[:, 0]).detach().cpu().numpy()
    return score_sup, score_q


# ============================================================
# 7. Episode evaluation
# ============================================================

def choose_support_query(
    y: np.ndarray,
    k: int,
    repeat: int,
    seed: int,
    query_cap_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1009 * repeat + 9176 * k)
    support_idx: List[int] = []
    query_idx: List[int] = []
    for c in [0, 1]:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            raise RuntimeError(f"No samples for class {c}")
        rng.shuffle(idx)
        if idx.size >= k:
            sup = idx[:k]
            q = idx[k:]
        else:
            sup = rng.choice(idx, size=k, replace=True)
            q = idx
        if query_cap_per_class and query_cap_per_class > 0 and q.size > query_cap_per_class:
            q = rng.choice(q, size=query_cap_per_class, replace=False)
        support_idx.extend([int(i) for i in sup])
        query_idx.extend([int(i) for i in q])
    rng.shuffle(support_idx)
    rng.shuffle(query_idx)
    return np.asarray(support_idx, dtype=np.int64), np.asarray(query_idx, dtype=np.int64)


def make_row(
    base_info: Dict[str, object],
    method: str,
    y_query: np.ndarray,
    score_query: np.ndarray,
    support_other_score: Optional[np.ndarray] = None,
    use_tc_for_acc_f1: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, object]:
    m = metrics_from_scores(
        y_query,
        score_query,
        threshold=0.0,
        support_other_score=support_other_score,
        tc_quantile=(args.cos_tail_quantile if args is not None else 0.95),
    )
    if use_tc_for_acc_f1:
        m["acc"] = m.get("acc_tc", m["acc"])
        m["f1"] = m.get("f1_tc", m["f1"])
    row = dict(base_info)
    row["method"] = method
    for key in METRICS:
        row[key] = m.get(key, float("nan"))
    return row


def evaluate_episode(
    z_target: np.ndarray,
    y_target: np.ndarray,
    task: str,
    k: int,
    seed: int,
    repeat: int,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    sup_idx, query_idx = choose_support_query(
        y=y_target,
        k=k,
        repeat=repeat,
        seed=seed,
        query_cap_per_class=args.query_cap_per_class,
    )
    z_sup = z_target[sup_idx]
    y_sup = y_target[sup_idx]
    z_q = z_target[query_idx]
    y_q = y_target[query_idx]

    base_info = {
        "task": task,
        "k_shot": k,
        "seed": seed,
        "repeat": repeat,
        "feature_mode": args.feature_mode,
        "src_tail_weight": args.src_tail_weight,
    }

    rows: List[Dict[str, object]] = []

    # CE-Proto
    anchors = []
    for c in [0, 1]:
        anchors.append(z_sup[y_sup == c].mean(axis=0))
    anchors_np = np.stack(anchors, axis=0).astype(np.float32)
    score_sup_proto = cosine_score_np(z_sup, anchors_np)
    score_q_proto = cosine_score_np(z_q, anchors_np)
    rows.append(make_row(base_info, "CE-Proto", y_q, score_q_proto, None, False, args))

    # CE-LinearHead
    score_sup_lin, score_q_lin = train_linear_head(z_sup, y_sup, z_q, args, device)
    rows.append(make_row(base_info, "CE-LinearHead", y_q, score_q_lin, None, False, args))

    # CosMetricHead without tail
    score_sup_cos, score_q_cos = train_cos_head(z_sup, y_sup, z_q, args, device, use_tail=False)
    other_sup_cos = score_sup_cos[y_sup == 0]
    rows.append(make_row(base_info, "CosMetricHead", y_q, score_q_cos, None, False, args))
    rows.append(make_row(base_info, "CosMetricHead-TC", y_q, score_q_cos, other_sup_cos, True, args))

    # CosMetricHead with target tail
    score_sup_tail, score_q_tail = train_cos_head(z_sup, y_sup, z_q, args, device, use_tail=True)
    other_sup_tail = score_sup_tail[y_sup == 0]
    rows.append(make_row(base_info, "CosMetricHead-Tail", y_q, score_q_tail, None, False, args))
    rows.append(make_row(base_info, "CosMetricHead-Tail-TC", y_q, score_q_tail, other_sup_tail, True, args))

    # Calibrated proxy-cosine metric head: cosine proxies + learnable bias/scale.
    # This is the key metric-learning check: can a proxy metric head absorb the
    # useful calibration of LinearHead without abandoning cosine distance?
    score_sup_calib, score_q_calib = train_calibrated_proxy_cos_head(
        z_sup, y_sup, z_q, args, device, use_tail=False
    )
    other_sup_calib = score_sup_calib[y_sup == 0]
    rows.append(make_row(base_info, "CalibCosMetricHead", y_q, score_q_calib, None, False, args))
    rows.append(make_row(base_info, "CalibCosMetricHead-TC", y_q, score_q_calib, other_sup_calib, True, args))

    score_sup_calib_tail, score_q_calib_tail = train_calibrated_proxy_cos_head(
        z_sup, y_sup, z_q, args, device, use_tail=True
    )
    other_sup_calib_tail = score_sup_calib_tail[y_sup == 0]
    rows.append(make_row(base_info, "CalibCosMetricHead-Tail", y_q, score_q_calib_tail, None, False, args))
    rows.append(make_row(base_info, "CalibCosMetricHead-Tail-TC", y_q, score_q_calib_tail, other_sup_calib_tail, True, args))

    # Norm-aware calibrated proxy-cosine metric head.
    # This tests whether the useful norm/magnitude information implicitly used by
    # LinearHead can be injected into a proxy-cosine metric model.
    score_sup_na, score_q_na = train_normaware_calibrated_proxy_cos_head(
        z_sup, y_sup, z_q, args, device, use_tail=False
    )
    other_sup_na = score_sup_na[y_sup == 0]
    rows.append(make_row(base_info, "NormAwareCalibCosMetricHead", y_q, score_q_na, None, False, args))
    rows.append(make_row(base_info, "NormAwareCalibCosMetricHead-TC", y_q, score_q_na, other_sup_na, True, args))

    score_sup_na_tail, score_q_na_tail = train_normaware_calibrated_proxy_cos_head(
        z_sup, y_sup, z_q, args, device, use_tail=True
    )
    other_sup_na_tail = score_sup_na_tail[y_sup == 0]
    rows.append(make_row(base_info, "NormAwareCalibCosMetricHead-Tail", y_q, score_q_na_tail, None, False, args))
    rows.append(make_row(base_info, "NormAwareCalibCosMetricHead-Tail-TC", y_q, score_q_na_tail, other_sup_na_tail, True, args))

    # Regularized diagonal Mahalanobis proxy-distance head.
    # This is the key check: can a true proxy-distance metric absorb LinearHead's
    # expressive advantage while keeping a K-shot distance-to-proxy interpretation?
    score_sup_maha, score_q_maha = train_reg_maha_proxy_head(
        z_sup, y_sup, z_q, args, device, use_tail=False
    )
    other_sup_maha = score_sup_maha[y_sup == 0]
    rows.append(make_row(base_info, "RegMahaProxy", y_q, score_q_maha, None, False, args))
    rows.append(make_row(base_info, "RegMahaProxy-TC", y_q, score_q_maha, other_sup_maha, True, args))

    score_sup_maha_tail, score_q_maha_tail = train_reg_maha_proxy_head(
        z_sup, y_sup, z_q, args, device, use_tail=True
    )
    other_sup_maha_tail = score_sup_maha_tail[y_sup == 0]
    rows.append(make_row(base_info, "RegMahaProxy-Tail", y_q, score_q_maha_tail, None, False, args))
    rows.append(make_row(base_info, "RegMahaProxy-Tail-TC", y_q, score_q_maha_tail, other_sup_maha_tail, True, args))


    # Linear-equivalent proxy distance head.
    # This is the strict test: can the strongest LinearHead be represented as a
    # proxy-distance metric and then improved by tail adaptation?
    score_sup_lineq, score_q_lineq = train_lineq_proxy_head(
        z_sup, y_sup, z_q, args, device, use_tail=False
    )
    other_sup_lineq = score_sup_lineq[y_sup == 0]
    rows.append(make_row(base_info, "LinEqProxy", y_q, score_q_lineq, None, False, args))
    rows.append(make_row(base_info, "LinEqProxy-TC", y_q, score_q_lineq, other_sup_lineq, True, args))

    score_sup_lineq_tail, score_q_lineq_tail = train_lineq_proxy_head(
        z_sup, y_sup, z_q, args, device, use_tail=True
    )
    other_sup_lineq_tail = score_sup_lineq_tail[y_sup == 0]
    rows.append(make_row(base_info, "LinEqProxy-Tail", y_q, score_q_lineq_tail, None, False, args))
    rows.append(make_row(base_info, "LinEqProxy-Tail-TC", y_q, score_q_lineq_tail, other_sup_lineq_tail, True, args))

    return rows


# ============================================================
# 8. Summary tables
# ============================================================

def summarize_results(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [m for m in METRICS if m in df.columns]

    # Mean/std by task/K/method.
    group_cols = ["task", "k_shot", "method"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    flat_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            flat_cols.append("_".join([x for x in c if x]))
        else:
            flat_cols.append(str(c))
    agg.columns = flat_cols
    agg.to_csv(out_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    # Deltas vs CE-Proto by K and method.
    keys = ["task", "k_shot", "seed", "repeat"]
    base = df[df["method"] == "CE-Proto"][keys + metric_cols].copy()
    delta_rows = []
    for method in sorted(df["method"].dropna().unique()):
        if method == "CE-Proto":
            continue
        cur = df[df["method"] == method][keys + metric_cols].copy()
        merged = base.merge(cur, on=keys, suffixes=("_base", "_new"))
        if merged.empty:
            continue
        for (task_v, k_v), g in merged.groupby(["task", "k_shot"]):
            row = {"task": task_v, "k_shot": k_v, "method": method, "baseline": "CE-Proto", "n": int(g.shape[0])}
            for m in metric_cols:
                d = (g[f"{m}_new"].astype(float) - g[f"{m}_base"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
                if d.empty:
                    row[f"{m}_delta_mean"] = float("nan")
                    row[f"{m}_delta_positive_ratio"] = float("nan")
                else:
                    row[f"{m}_delta_mean"] = float(d.mean())
                    row[f"{m}_delta_positive_ratio"] = float((d > 0).mean())
            delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "delta_by_k.csv", index=False, encoding="utf-8-sig")

    # Overall verdict vs CE-Proto.
    overall_rows = []
    for method in sorted(df["method"].dropna().unique()):
        if method == "CE-Proto":
            continue
        cur = df[df["method"] == method][keys + metric_cols].copy()
        merged = base.merge(cur, on=keys, suffixes=("_base", "_new"))
        if merged.empty:
            continue
        row = {"method": method, "baseline": "CE-Proto", "n": int(merged.shape[0])}
        for m in metric_cols:
            d = (merged[f"{m}_new"].astype(float) - merged[f"{m}_base"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                row[f"{m}_delta_mean"] = float("nan")
                row[f"{m}_delta_positive_ratio"] = float("nan")
            else:
                row[f"{m}_delta_mean"] = float(d.mean())
                row[f"{m}_delta_positive_ratio"] = float((d > 0).mean())

        auc_d = row.get("auc_delta_mean", float("nan"))
        pd_d = row.get("pd_at_pfa_0.05_delta_mean", float("nan"))
        auc_r = row.get("auc_delta_positive_ratio", 0.0)
        pd_r = row.get("pd_at_pfa_0.05_delta_positive_ratio", 0.0)
        if np.isfinite(auc_d) and np.isfinite(pd_d) and auc_d > 0 and pd_d > 0 and auc_r >= 0.6 and pd_r >= 0.6:
            judgement = "supported"
        elif np.isfinite(auc_d) and np.isfinite(pd_d) and auc_d > 0 and pd_d > 0:
            judgement = "weak_support"
        else:
            judgement = "not_supported"
        row["judgement"] = judgement
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_dir / "overall_verdict.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 9. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data and task.
    ap.add_argument("--data_root", type=str, default="data_targets")
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="2026")
    ap.add_argument("--k_shots", type=str, default="5,10,20")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--query_cap_per_class", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="outputs_dt_feature_cmtc")
    ap.add_argument("--cache_dir", type=str, default="cache_dt_feature_cmtc")
    ap.add_argument("--rebuild_cache", action="store_true")

    # Audio/feature params.
    ap.add_argument("--target_sr", type=int, default=12000)
    ap.add_argument("--segment_sec", type=float, default=4.0)
    ap.add_argument("--segment_hop_sec", type=float, default=4.0)
    ap.add_argument("--feature_mode", type=str, default="stft", choices=["stft", "logmel", "mhp"])
    ap.add_argument("--feature_norm", type=str, default="sample", choices=["sample", "global", "none"],
                    help="sample=per-segment norm; global=source-train stats; none=no norm")
    ap.add_argument("--log_power_mode", type=str, default="log1p", choices=["log1p", "db"],
                    help="db roughly matches librosa.power_to_db(..., top_db=None)")
    ap.add_argument("--stft_window", type=str, default="hamming", choices=["hamming", "hann"],
                    help="paper code used np.hanning; hann is closest")
    ap.add_argument("--single_to_rgb", action="store_true",
                    help="repeat single-channel stft/logmel to 3 channels, like the uploaded paper code")
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--feature_time_bins", type=int, default=96)
    ap.add_argument("--stft_freq_bins", type=int, default=128)
    ap.add_argument("--mel_n_mels", type=int, default=128)
    ap.add_argument("--mel_fmin", type=float, default=10.0)
    ap.add_argument("--mel_fmax", type=float, default=800.0)
    ap.add_argument("--hpss_harmonic_kernel", type=int, default=17)
    ap.add_argument("--hpss_percussive_kernel", type=int, default=17)

    # Training params.
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--backbone", type=str, default="small", choices=["small", "resnet18"],
                    help="small = current lightweight CNN; resnet18 = closer to uploaded paper code")
    ap.add_argument("--encoder_width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--epochs_ce", type=int, default=10)
    ap.add_argument("--epochs_hpmd", type=int, default=0, help="Accepted for compatibility; unused.")
    ap.add_argument("--lr_ce", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--linear_epochs", type=int, default=100)
    ap.add_argument("--lr_linear", type=float, default=1e-2)
    ap.add_argument("--cos_epochs", type=int, default=120)
    ap.add_argument("--lr_cos", type=float, default=5e-2)
    ap.add_argument("--cos_scale", type=float, default=10.0)

    # Calibrated proxy-cosine target head.
    ap.add_argument("--calib_cos_epochs", type=int, default=120)
    ap.add_argument("--lr_calib_cos", type=float, default=5e-2)
    ap.add_argument("--calib_cos_scale", type=float, default=10.0)
    ap.add_argument("--calib_cos_fixed_scale", action="store_true")

    # Norm-aware calibrated proxy-cosine target head.
    ap.add_argument("--na_calib_cos_epochs", type=int, default=120)
    ap.add_argument("--lr_na_calib_cos", type=float, default=5e-2)
    ap.add_argument("--na_calib_cos_scale", type=float, default=10.0)
    ap.add_argument("--na_calib_cos_fixed_scale", action="store_true")
    ap.add_argument("--na_calib_init_gamma", type=float, default=0.0)
    ap.add_argument("--na_calib_gamma_lr_mult", type=float, default=0.25,
                    help="LR multiplier for norm-aware gamma parameters; smaller is usually more stable")
    ap.add_argument("--na_calib_gamma_l2", type=float, default=1e-4,
                    help="Small L2 penalty on gamma to prevent overfitting K-shot support")

    # Regularized Mahalanobis proxy-distance target head.
    ap.add_argument("--maha_epochs", type=int, default=120)
    ap.add_argument("--lr_maha", type=float, default=5e-2)
    ap.add_argument("--maha_scale", type=float, default=10.0)
    ap.add_argument("--maha_fixed_scale", action="store_true")
    ap.add_argument("--maha_proxy_l2", type=float, default=1e-3,
                    help="Anchor learned proxies near support class means.")
    ap.add_argument("--maha_diag_l2", type=float, default=1e-3,
                    help="Keep diagonal Mahalanobis weights near identity.")
    ap.add_argument("--maha_bias_l2", type=float, default=0.0,
                    help="Optional L2 penalty on class biases.")

    # Target tail.

    # Linear-equivalent proxy distance head
    ap.add_argument("--lineq_epochs", type=int, default=120,
                    help="Training epochs for LinEqProxy-Tail / LinEqProxy-Tail-TC.")
    ap.add_argument("--lineq_finetune_epochs", type=int, default=0,
                    help="Optional fine-tuning epochs for LinEqProxy without Tail. 0 keeps the exact LinearHead-equivalent proxy.")
    ap.add_argument("--lr_lineq", type=float, default=5e-2,
                    help="Learning rate for LinEqProxy fine-tuning.")
    ap.add_argument("--lineq_proxy_l2", type=float, default=1e-3,
                    help="L2 regularization around initial Linear-equivalent proxies.")
    ap.add_argument("--lineq_bias_l2", type=float, default=1e-4,
                    help="L2 regularization around initial Linear-equivalent bias.")
    ap.add_argument("--cos_tail_weight", type=float, default=0.5)
    ap.add_argument("--cos_tail_margin", type=float, default=0.10)
    ap.add_argument("--cos_tail_quantile", type=float, default=0.95)

    # Source tail.
    ap.add_argument("--src_tail_weight", type=float, default=0.0)
    ap.add_argument("--src_tail_margin", type=float, default=0.10)
    ap.add_argument("--src_tail_quantile", type=float, default=0.95)
    ap.add_argument("--src_tail_warmup_epochs", type=int, default=3)

    # Source pAUC pairwise ranking loss, adapted from uploaded xuan_loss.py.
    ap.add_argument("--src_pauc_weight", type=float, default=0.0)
    ap.add_argument("--src_pauc_alpha", type=float, default=0.00)
    ap.add_argument("--src_pauc_beta", type=float, default=0.10)
    ap.add_argument("--src_pauc_margin", type=float, default=0.10)
    ap.add_argument("--src_pauc_warmup_epochs", type=int, default=3)

    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(Path(args.cache_dir))

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[{now()}] device={device}")
    print(f"feature_mode={args.feature_mode}, feature_norm={args.feature_norm}, log_power_mode={args.log_power_mode}, single_to_rgb={args.single_to_rgb}")
    print(f"backbone={args.backbone}, in_channels={infer_in_channels(args.feature_mode, args.single_to_rgb)}")
    print(f"src_tail_weight={args.src_tail_weight}, src_pauc_weight={args.src_pauc_weight}, target_tail_weight={args.cos_tail_weight}")

    tasks = parse_tasks(args.tasks)
    seeds = parse_int_list(args.seeds)
    k_shots = parse_int_list(args.k_shots)
    data_root = Path(args.data_root)

    # Save run config.
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    all_rows: List[Dict[str, object]] = []

    for src_domains, tgt_domain in tasks:
        tname = task_name(src_domains, tgt_domain)
        print("\n" + "=" * 80)
        print(f"TASK {tname}")
        print("=" * 80)

        # Load features.
        source_Xs = []
        source_ys = []
        for d in src_domains:
            Xd, yd = load_domain_features(data_root, d, args)
            source_Xs.append(Xd)
            source_ys.append(yd)
        X_src = np.concatenate(source_Xs, axis=0)
        y_src = np.concatenate(source_ys, axis=0)
        X_tgt, y_tgt = load_domain_features(data_root, tgt_domain, args)

        if args.feature_norm == "global":
            X_src, X_tgt = apply_global_feature_norm(X_src, X_tgt)

        X_src = maybe_replicate_single_to_rgb(X_src, args)
        X_tgt = maybe_replicate_single_to_rgb(X_tgt, args)

        print(f"[SOURCE] X={X_src.shape}, other={int((y_src==0).sum())}, target={int((y_src==1).sum())}")
        print(f"[TARGET] X={X_tgt.shape}, other={int((y_tgt==0).sum())}, target={int((y_tgt==1).sum())}")

        for seed in seeds:
            print(f"\n[SEED] {seed} train source encoder")
            encoder, _ = train_source_encoder(X_src, y_src, args, device, seed=seed)
            z_tgt = encode_array(encoder, X_tgt, device=device, batch_size=max(64, args.batch_size * 8))

            for k in k_shots:
                for rep in range(args.repeats):
                    rows = evaluate_episode(
                        z_target=z_tgt,
                        y_target=y_tgt,
                        task=tname,
                        k=k,
                        seed=seed,
                        repeat=rep,
                        args=args,
                        device=device,
                    )
                    all_rows.extend(rows)

                    if len(all_rows) % 60 == 0:
                        pd.DataFrame(all_rows).to_csv(out_dir / "all_metrics_partial.csv", index=False, encoding="utf-8-sig")
                print(f"  [DONE] task={tname} seed={seed} K={k}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    summarize_results(df, out_dir)

    print("\n[OK] saved:")
    print(out_dir / "all_metrics.csv")
    print(out_dir / "summary_mean_std.csv")
    print(out_dir / "delta_by_k.csv")
    print(out_dir / "overall_verdict.csv")


if __name__ == "__main__":
    main()
