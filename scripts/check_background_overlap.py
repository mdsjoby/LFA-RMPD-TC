# -*- coding: utf-8 -*-
"""
Check cross-domain filename/hash overlap in the local data_targets manifest.

Run audit_data_targets.py first, then run this script on the generated manifest.
The main purpose is to make source/target background reuse explicit.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def parse_tasks(text: str) -> List[Tuple[List[str], str]]:
    tasks: List[Tuple[List[str], str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Task must be formatted like open+speedboat:uuv, got {item!r}")
        src, tgt = item.split(":", 1)
        src_domains = [x.strip() for x in src.split("+") if x.strip()]
        tgt_domain = tgt.strip()
        if not src_domains or not tgt_domain:
            raise ValueError(f"Invalid task: {item!r}")
        tasks.append((src_domains, tgt_domain))
    return tasks


def task_name(src_domains: Sequence[str], target_domain: str) -> str:
    return "+".join(src_domains) + "->" + target_domain


def normalize_rel_pattern(relative_path: str) -> str:
    text = str(relative_path).replace("\\", "/").lower()
    text = re.sub(r"^[^/]+/", "", text)
    text = re.sub(r"^(raw/)?(target|other|background|positive|negative|pos|neg|0|1)/", "", text)
    return text


def pair_rows_for_key(df: pd.DataFrame, key: str, pair_type: str, max_rows: int = 200000) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    valid = df[df[key].notna()].copy()
    for value, g in valid.groupby(key, sort=False):
        if len(g) < 2 or g["domain"].nunique() < 2:
            continue
        records = g.to_dict("records")
        for a, b in itertools.combinations(records, 2):
            if a["domain"] == b["domain"]:
                continue
            rows.append({
                "pair_type": pair_type,
                "key": value,
                "domain_a": a["domain"],
                "class_a": a["class_label"],
                "relative_path_a": a["relative_path"],
                "sha256_a": a.get("sha256", ""),
                "domain_b": b["domain"],
                "class_b": b["class_label"],
                "relative_path_b": b["relative_path"],
                "sha256_b": b.get("sha256", ""),
            })
            if max_rows > 0 and len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def overlap_count(a: pd.DataFrame, b: pd.DataFrame, key: str) -> int:
    if a.empty or b.empty:
        return 0
    return int(len(set(a[key]).intersection(set(b[key]))))


def overlap_rows_by_task(df: pd.DataFrame, tasks: Iterable[Tuple[List[str], str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    comparisons = [
        ("source_target_vs_target_target", "target", "target"),
        ("source_other_vs_target_other", "other", "other"),
        ("source_all_vs_target_all", None, None),
        ("source_all_vs_target_other", None, "other"),
        ("source_all_vs_target_target", None, "target"),
    ]
    for src_domains, tgt_domain in tasks:
        tname = task_name(src_domains, tgt_domain)
        source_all = df[df["domain"].isin(src_domains)]
        target_all = df[df["domain"] == tgt_domain]
        for comp, src_class, tgt_class in comparisons:
            src = source_all if src_class is None else source_all[source_all["class_label"] == src_class]
            tgt = target_all if tgt_class is None else target_all[target_all["class_label"] == tgt_class]
            hash_overlap = overlap_count(src, tgt, "sha256")
            basename_overlap = overlap_count(src, tgt, "basename")
            pattern_overlap = overlap_count(src, tgt, "relative_pattern")
            duration_size_overlap = overlap_count(src, tgt, "duration_size_key")
            rows.append({
                "task": tname,
                "comparison": comp,
                "source_domains": "+".join(src_domains),
                "target_domain": tgt_domain,
                "source_class": src_class if src_class is not None else "all",
                "target_class": tgt_class if tgt_class is not None else "all",
                "source_count": int(len(src)),
                "target_count": int(len(tgt)),
                "sha256_overlap_count": hash_overlap,
                "sha256_target_overlap_ratio": hash_overlap / max(int(len(tgt)), 1),
                "sha256_source_overlap_ratio": hash_overlap / max(int(len(src)), 1),
                "basename_overlap_count": basename_overlap,
                "relative_pattern_overlap_count": pattern_overlap,
                "duration_size_overlap_count": duration_size_overlap,
            })
    return pd.DataFrame(rows)


def summarize_text(
    df: pd.DataFrame,
    hash_pairs: pd.DataFrame,
    basename_pairs: pd.DataFrame,
    task_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("Cross-domain data_targets overlap audit")
    lines.append("=" * 43)
    lines.append("")
    lines.append(f"Total manifest records: {len(df)}")
    lines.append(f"Unique SHA256 hashes: {df['sha256'].nunique()}")
    lines.append(f"Cross-domain identical SHA256 pairs: {len(hash_pairs)}")
    lines.append(f"Cross-domain same-basename pairs: {len(basename_pairs)}")
    lines.append("")

    if hash_pairs.empty:
        lines.append("No identical WAV SHA256 overlap was detected across domains.")
    else:
        other_pairs = hash_pairs[(hash_pairs["class_a"] == "other") & (hash_pairs["class_b"] == "other")]
        target_pairs = hash_pairs[(hash_pairs["class_a"] == "target") | (hash_pairs["class_b"] == "target")]
        lines.append(
            f"Identical SHA256 overlap is present: other/background-only pairs={len(other_pairs)}, "
            f"pairs involving at least one target record={len(target_pairs)}."
        )
        if len(target_pairs) == 0 and len(other_pairs) > 0:
            lines.append(
                "The detected identical overlap is confined to background/other records in this manifest."
            )
        else:
            lines.append(
                "Some identical overlap involves target-labelled records; inspect duplicate_hash_pairs.csv before reporting results."
            )
    lines.append("")
    lines.append("Task-level overlap:")
    for _, row in task_summary.iterrows():
        if row["comparison"] in {"source_target_vs_target_target", "source_other_vs_target_other", "source_all_vs_target_all"}:
            lines.append(
                "- {task} | {comparison}: hash={sha256_overlap_count}/{target_count} target-side "
                "({sha256_target_overlap_ratio:.3%}), basename={basename_overlap_count}, "
                "pattern={relative_pattern_overlap_count}".format(**row.to_dict())
            )
    lines.append("")
    lines.append("Recommended manuscript wording:")
    if hash_pairs.empty:
        lines.append(
            "A SHA256-based audit found no identical WAV files shared between source and target-domain records for the evaluated tasks."
        )
    else:
        lines.append(
            "A SHA256-based audit found that the public domain folders share some identical background/other WAV records across domains; "
            "therefore the main protocol is reported as an episode-level cross-target evaluation rather than a strict deployment-level "
            "recording-independent split. A hash-disjoint robustness check removes target-domain records whose hashes appear in the source domains."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="outputs_audit/data_targets_manifest.csv")
    ap.add_argument("--out_dir", type=str, default="outputs_audit")
    ap.add_argument("--tasks", type=str, default="open+speedboat:uuv,open+uuv:speedboat")
    ap.add_argument("--max_pair_rows", type=int, default=200000)
    ap.add_argument("--max_duration_size_pair_rows", type=int, default=50000)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"domain", "class_label", "relative_path", "basename", "sha256", "duration_sec", "file_size"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    df = df.copy()
    df["relative_pattern"] = df["relative_path"].map(normalize_rel_pattern)
    df["duration_size_key"] = (
        df["duration_sec"].round(6).astype(str) + "|" + df["file_size"].astype(str)
    )

    hash_pairs = pair_rows_for_key(df, "sha256", "sha256", max_rows=int(args.max_pair_rows))
    basename_pairs = pair_rows_for_key(df, "basename", "basename", max_rows=int(args.max_pair_rows))
    pattern_pairs = pair_rows_for_key(df, "relative_pattern", "relative_pattern", max_rows=int(args.max_pair_rows))
    duration_size_pairs = pair_rows_for_key(
        df,
        "duration_size_key",
        "duration_size",
        max_rows=int(args.max_duration_size_pair_rows),
    )
    task_summary = overlap_rows_by_task(df, parse_tasks(args.tasks))

    hash_pairs.to_csv(out_dir / "duplicate_hash_pairs.csv", index=False, encoding="utf-8-sig")
    basename_pairs.to_csv(out_dir / "duplicate_basename_pairs.csv", index=False, encoding="utf-8-sig")
    pattern_pairs.to_csv(out_dir / "duplicate_relative_pattern_pairs.csv", index=False, encoding="utf-8-sig")
    duration_size_pairs.to_csv(out_dir / "duplicate_duration_size_pairs.csv", index=False, encoding="utf-8-sig")
    task_summary.to_csv(out_dir / "background_overlap_by_task.csv", index=False, encoding="utf-8-sig")
    summarize_text(df, hash_pairs, basename_pairs, task_summary, out_dir / "overlap_summary.txt")

    print(f"Wrote {out_dir / 'duplicate_hash_pairs.csv'}")
    print(f"Wrote {out_dir / 'background_overlap_by_task.csv'}")
    print(f"Wrote {out_dir / 'overlap_summary.txt'}")


if __name__ == "__main__":
    main()
