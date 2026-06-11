# -*- coding: utf-8 -*-
"""
Generate QiandaoEar22 acoustic explanation figures for the Applied Acoustics
manuscript.

Run from the project root, e.g.

    python plot_qiandaoear22_logmel_examples.py ^
      --data_root data_targets ^
      --out_dir outputs_acoustic_figures

The script uses the same formal log-mel settings used in the manuscript:
12 kHz sampling rate, 4 s model input, 1024-point Hann STFT, hop length 512,
128 mel bands, 20-800 Hz, and 96 time bins. The figure itself crops pure
zero-padding columns for readability; the formal model input remains 4 s.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fmax = min(float(fmax), sr / 2.0)
    mel_points = np.linspace(hz_to_mel(np.asarray([fmin]))[0], hz_to_mel(np.asarray([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def read_wav_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    sr, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return x, sr


def segment_first(x: np.ndarray, sr: int, seg_sec: float) -> np.ndarray:
    seg_len = int(round(seg_sec * sr))
    if len(x) < seg_len:
        x = np.pad(x, (0, seg_len - len(x)))
    return x[:seg_len].astype(np.float32)


def logmel_db(
    seg: np.ndarray,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    out_t: int,
) -> np.ndarray:
    _, _, z = signal.stft(
        seg,
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = (np.abs(z) ** 2).astype(np.float32)
    fb = build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = fb @ power
    mel = np.maximum(mel, 1e-10)
    feat = 10.0 * np.log10(mel)
    if feat.shape[1] >= out_t:
        feat = feat[:, :out_t]
    else:
        feat = np.pad(feat, ((0, 0), (0, out_t - feat.shape[1])), constant_values=-100.0)
    return feat.astype(np.float32)


def list_wavs(domain_dir: Path, label_name: str) -> List[Path]:
    root = domain_dir / "raw" / label_name
    if not root.exists():
        root = domain_dir / label_name
    return sorted(root.rglob("*.wav"))


def choose_example(files: List[Path], strategy: str, index: int) -> Path:
    if not files:
        raise RuntimeError("No WAV files available for example selection.")
    if strategy == "first":
        return files[0]
    if strategy == "last":
        return files[-1]
    if strategy == "index":
        return files[max(0, min(index, len(files) - 1))]
    # default: middle, deterministic and usually not a boundary case.
    return files[len(files) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_targets")
    ap.add_argument("--out_dir", type=str, default="outputs_acoustic_figures")
    ap.add_argument("--target_sr", type=int, default=12000)
    ap.add_argument("--seg_sec", type=float, default=4.0)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--mel_fmin", type=float, default=20.0)
    ap.add_argument("--mel_fmax", type=float, default=800.0)
    ap.add_argument("--time_bins", type=int, default=96)
    ap.add_argument("--example_strategy", choices=["middle", "first", "last", "index"], default="middle")
    ap.add_argument("--example_index", type=int, default=0)
    ap.add_argument("--show_padding", action="store_true", help="Show full 4 s model input including padded columns.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = [
        ("open", "KaiYuan"),
        ("speedboat", "SpeedBoat"),
        ("uuv", "UUV"),
    ]
    labels = [
        ("other", "other/background"),
        ("target", "target"),
    ]

    fig, axes = plt.subplots(len(domains), len(labels), figsize=(8.4, 7.0), dpi=180, sharey=True)
    used: List[Dict[str, object]] = []

    for r, (domain, official_name) in enumerate(domains):
        domain_dir = data_root / domain
        for c, (folder, class_name) in enumerate(labels):
            files = list_wavs(domain_dir, folder)
            p = choose_example(files, args.example_strategy, args.example_index)
            wav, sr = read_wav_mono(p, args.target_sr)
            duration = min(len(wav) / sr, args.seg_sec)
            seg = segment_first(wav, sr, args.seg_sec)
            feat = logmel_db(
                seg,
                sr=sr,
                n_fft=args.n_fft,
                win_length=args.win_length,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmin=args.mel_fmin,
                fmax=args.mel_fmax,
                out_t=args.time_bins,
            )
            plot_feat = feat
            plot_duration = args.seg_sec
            if not args.show_padding:
                # Hide pure padding columns in the visualization only. This
                # does not alter model inputs used by experiments.
                valid_cols = np.where(plot_feat.max(axis=0) > -95.0)[0]
                if len(valid_cols):
                    plot_feat = plot_feat[:, :valid_cols[-1] + 1]
                    plot_duration = duration

            ax = axes[r, c]
            im = ax.imshow(
                plot_feat,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=[0, plot_duration, args.mel_fmin, args.mel_fmax],
                vmin=-100,
                vmax=-20,
            )
            ax.set_title(f"{official_name} {class_name}", fontsize=10)
            if c == 0:
                ax.set_ylabel("Frequency (Hz)")
            if r == len(domains) - 1:
                ax.set_xlabel("Time (s)")
            used.append({
                "local_domain": domain,
                "official_category": official_name,
                "class": class_name,
                "duration_s": round(float(duration), 3),
                "relative_path": str(p.relative_to(data_root)),
            })

    fig.subplots_adjust(right=0.88, hspace=0.38, wspace=0.18)
    cbar_ax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="Log-mel power (dB-like)")

    fig_path = out_dir / "qiandaoear22_logmel_examples.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    used_path = out_dir / "qiandaoear22_logmel_examples_files.csv"
    pd.DataFrame(used).to_csv(used_path, index=False, encoding="utf-8-sig")

    caption_path = out_dir / "qiandaoear22_logmel_examples_caption.txt"
    caption_path.write_text(
        "Fig. X. Representative log-mel examples from the QiandaoEar22 KaiYuan, "
        "SpeedBoat, and UUV subdatasets. Each row shows one target-specific "
        "domain, and columns show other/background and target examples. The "
        "visualization crops pure zero-padding columns for readability; the "
        "formal model input uses 4 s segments with 96 time bins.\\n",
        encoding="utf-8",
    )

    print(f"[OK] figure: {fig_path}")
    print(f"[OK] file list: {used_path}")
    print(f"[OK] caption: {caption_path}")


if __name__ == "__main__":
    main()
