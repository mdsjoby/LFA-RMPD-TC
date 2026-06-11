# -*- coding: utf-8 -*-
"""
Audit the local QiandaoEar22-style data_targets directory.

The script creates a record-level WAV manifest with duration and hash metadata.
It intentionally reports local preprocessed WAV records, not derived model
segments, because the model later maps each record to a fixed-length input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


AUDIO_EXTS = {".wav", ".wave"}
TARGET_ALIASES = {"target", "positive", "pos", "1"}
OTHER_ALIASES = {"other", "background", "bg", "negative", "neg", "0"}


def parse_domains(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def classify_file(path: Path, domain_dir: Path) -> Optional[str]:
    try:
        rel_parts = [p.lower() for p in path.relative_to(domain_dir).parts]
    except ValueError:
        rel_parts = [p.lower() for p in path.parts]
    if any(p in TARGET_ALIASES for p in rel_parts):
        return "target"
    if any(p in OTHER_ALIASES for p in rel_parts):
        return "other"
    text = "/".join(rel_parts)
    if any(tok in text for tok in ["target", "positive", "pos"]):
        return "target"
    if any(tok in text for tok in ["other", "background", "negative", "neg"]):
        return "other"
    return None


def wav_meta(path: Path) -> Dict[str, object]:
    with wave.open(str(path), "rb") as wf:
        frames = int(wf.getnframes())
        sample_rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "sample_width": sample_width,
        "duration_sec": duration,
    }


def iter_domain_records(data_root: Path, domains: Iterable[str]) -> Iterable[Dict[str, object]]:
    for domain in domains:
        domain_dir = data_root / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
        files = sorted(p for p in domain_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
        for path in files:
            class_label = classify_file(path, domain_dir)
            if class_label is None:
                continue
            try:
                meta = wav_meta(path)
                digest = sha256_file(path)
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}")
                continue
            rel = path.relative_to(data_root)
            row: Dict[str, object] = {
                "domain": domain,
                "class_label": class_label,
                "label": 1 if class_label == "target" else 0,
                "absolute_path": str(path.resolve()),
                "relative_path": rel.as_posix(),
                "basename": path.name,
                "stem": path.stem,
                "file_size": int(path.stat().st_size),
                "sha256": digest,
                "content_hash": digest,
            }
            row.update(meta)
            yield row


def safe_stat(values: List[float], name: str) -> Dict[str, float]:
    if not values:
        return {
            f"{name}_min": float("nan"),
            f"{name}_max": float("nan"),
            f"{name}_mean": float("nan"),
            f"{name}_median": float("nan"),
        }
    return {
        f"{name}_min": float(min(values)),
        f"{name}_max": float(max(values)),
        f"{name}_mean": float(statistics.mean(values)),
        f"{name}_median": float(statistics.median(values)),
    }


def build_summary(df: pd.DataFrame, segment_sec: float, approx_3_tol: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for domain, g in df.groupby("domain", sort=True):
        durations = [float(x) for x in g["duration_sec"].dropna().tolist()]
        row: Dict[str, object] = {
            "domain": domain,
            "target_count": int((g["class_label"] == "target").sum()),
            "other_count": int((g["class_label"] == "other").sum()),
            "total_count": int(len(g)),
            "total_duration_sec": float(sum(durations)),
            "total_duration_hours": float(sum(durations) / 3600.0),
            "shorter_than_model_input_count": int((g["duration_sec"] < segment_sec).sum()),
            "approx_3s_count": int(((g["duration_sec"] - 3.0).abs() <= approx_3_tol).sum()),
            "longer_than_model_input_count": int((g["duration_sec"] > segment_sec).sum()),
            "unique_sha256_count": int(g["sha256"].nunique()),
            "unique_basename_count": int(g["basename"].nunique()),
        }
        row.update(safe_stat(durations, "duration_sec"))
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_text(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path, segment_sec: float) -> None:
    lines: List[str] = []
    lines.append("QiandaoEar22 local data_targets audit")
    lines.append("=" * 44)
    lines.append("")
    lines.append("The reported counts refer to local preprocessed WAV records, not derived model segments.")
    lines.append(
        f"The model maps each local WAV record to a fixed {segment_sec:.1f}-s input by cropping or zero padding."
    )
    if int(summary["approx_3s_count"].sum()) == int(summary["total_count"].sum()):
        lines.append(
            "All detected local QiandaoEar22 records are approximately 3 s; therefore they are zero padded for the fixed model input."
        )
    elif int(summary["shorter_than_model_input_count"].sum()) > 0:
        lines.append(
            "Some records are shorter than the fixed model input and are zero padded during feature extraction."
        )
    lines.append("")
    total_count = int(summary["total_count"].sum())
    total_target = int(summary["target_count"].sum())
    total_other = int(summary["other_count"].sum())
    total_hours = float(summary["total_duration_hours"].sum())
    lines.append(f"Total records: {total_count}")
    lines.append(f"Target records: {total_target}")
    lines.append(f"Other/background records: {total_other}")
    lines.append(f"Total local WAV duration: {total_hours:.3f} h")
    lines.append("")
    lines.append("By domain:")
    for _, row in summary.sort_values("domain").iterrows():
        lines.append(
            "- {domain}: target={target_count}, other={other_count}, total={total_count}, "
            "duration mean={duration_sec_mean:.3f}s, min={duration_sec_min:.3f}s, max={duration_sec_max:.3f}s, "
            "shorter-than-input={shorter_than_model_input_count}".format(**row.to_dict())
        )
    lines.append("")
    lines.append("Class counts by domain:")
    counts = df.pivot_table(index="domain", columns="class_label", values="absolute_path", aggfunc="count", fill_value=0)
    lines.append(counts.to_string())
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_targets")
    ap.add_argument("--out_dir", type=str, default="outputs_audit")
    ap.add_argument("--domains", type=str, default="open,speedboat,uuv")
    ap.add_argument("--segment_sec", type=float, default=4.0)
    ap.add_argument("--approx_3_tol", type=float, default=0.05)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_domain_records(data_root, parse_domains(args.domains)))
    if not rows:
        raise RuntimeError(f"No labelled WAV files found under {data_root}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "data_targets_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = build_summary(manifest, segment_sec=float(args.segment_sec), approx_3_tol=float(args.approx_3_tol))
    summary_path = out_dir / "data_targets_summary_by_domain.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    text_path = out_dir / "data_targets_summary.txt"
    write_summary_text(manifest, summary, text_path, segment_sec=float(args.segment_sec))

    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()
