# -*- coding: utf-8 -*-
"""
Hash-disjoint robustness check for LFA-RMPD-TC.

This script reuses the formal log-mel feature settings and the method
implementations in run_lineq_proxy_pauc_cmtc.py. It differs from the main
experiment only in target-domain episode construction:

  1. Compute SHA256 hashes for all source-domain and target-domain WAV records.
  2. Remove overlapping records either from the target evaluation pool
     (--drop_side target) or from source training (--drop_side source).
  3. Build K-shot support/query episodes after the requested removal.

The formal main results should remain unchanged. This is a robustness audit for
the public QiandaoEar22 folder split, where identical background files can occur
across domain folders.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import random
import re
import shutil
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def load_lineq_module(project_root: Path):
    module_path = project_root / "run_lineq_proxy_pauc_cmtc.py"
    spec = importlib.util.spec_from_file_location("lineq_cmtc", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_tasks(text: str) -> List[Tuple[List[str], str]]:
    out: List[Tuple[List[str], str]] = []
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
        out.append((src_domains, tgt_domain))
    if not out:
        raise ValueError("No valid tasks.")
    return out


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def task_name(src_domains: Sequence[str], target_domain: str) -> str:
    return "+".join(src_domains) + "->" + target_domain


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = int(wf.getnframes())
        sample_rate = int(wf.getframerate())
    return frames / float(sample_rate) if sample_rate > 0 else 0.0


def expected_segment_count(duration_sec: float, target_sr: int, segment_sec: float, hop_sec: float) -> int:
    seg_len = max(1, int(round(segment_sec * target_sr)))
    hop_len = max(1, int(round(hop_sec * target_sr)))
    n = max(1, int(round(duration_sec * target_sr)))
    if n <= seg_len:
        return 1
    return max(1, int(math.floor((n - seg_len) / hop_len) + 1))


def infer_group_id(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_label__\d+__\d+(?: - .*)?$", "", stem)
    return stem


def read_manifest(manifest_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    by_rel: Dict[str, Dict[str, object]] = {}
    for row in df.to_dict("records"):
        rel = str(row.get("relative_path", "")).replace("\\", "/").lower()
        if rel:
            by_rel[rel] = row
    return by_rel


def file_records_for_domain(
    lineq,
    data_root: Path,
    domain: str,
    args,
    manifest_by_rel: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    label_files = lineq.find_label_files(data_root / domain)
    rows: List[Dict[str, object]] = []
    feature_index = 0
    for label in [0, 1]:
        for path in label_files[label]:
            rel = path.relative_to(data_root).as_posix()
            rel_key = rel.lower()
            cached = manifest_by_rel.get(rel_key, {})
            digest = str(cached.get("sha256") or sha256_file(path))
            duration = float(cached.get("duration_sec") or wav_duration(path))
            nseg = expected_segment_count(
                duration,
                target_sr=int(args.target_sr),
                segment_sec=float(args.segment_sec),
                hop_sec=float(args.segment_hop_sec),
            )
            for segment_index in range(nseg):
                rows.append({
                    "feature_index": feature_index,
                    "domain": domain,
                    "label": int(label),
                    "class_label": "target" if label == 1 else "other",
                    "absolute_path": str(path.resolve()),
                    "relative_path": rel,
                    "basename": path.name,
                    "sha256": digest,
                    "duration_sec": duration,
                    "segment_index": segment_index,
                    "segments_from_record": nseg,
                    "group_id": infer_group_id(path),
                })
                feature_index += 1
    return pd.DataFrame(rows)


def load_formal_args(lineq, project_root: Path, args0: argparse.Namespace):
    class Args:
        pass

    formal_cfg = project_root / args0.formal_out_dir / "run_config.txt"
    if not formal_cfg.exists():
        raise FileNotFoundError(f"Formal run_config not found: {formal_cfg}")

    cfg: Dict[str, str] = {}
    for line in formal_cfg.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            cfg[k] = v

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    int_fields = [
        "target_sr", "n_fft", "hop_length", "win_length", "feature_time_bins",
        "stft_freq_bins", "mel_n_mels", "batch_size", "emb_dim", "encoder_width",
        "epochs_ce", "linear_epochs", "cos_epochs", "maha_epochs",
        "src_pauc_warmup_epochs", "hpss_harmonic_kernel", "hpss_percussive_kernel",
    ]
    float_fields = [
        "segment_sec", "segment_hop_sec", "mel_fmin", "mel_fmax", "dropout",
        "lr_ce", "weight_decay", "lr_linear", "lr_cos", "cos_scale",
        "lr_maha", "maha_scale", "maha_proxy_l2", "maha_diag_l2",
        "maha_bias_l2", "cos_tail_weight", "cos_tail_margin",
        "cos_tail_quantile", "src_tail_weight", "src_pauc_weight",
        "src_pauc_alpha", "src_pauc_beta", "src_pauc_margin",
    ]
    bool_fields = [
        "single_to_rgb", "maha_fixed_scale", "calib_cos_fixed_scale",
        "na_calib_cos_fixed_scale", "verbose",
    ]
    for name in int_fields:
        if hasattr(args, name):
            setattr(args, name, int(getattr(args, name)))
    for name in float_fields:
        if hasattr(args, name):
            setattr(args, name, float(getattr(args, name)))
    for name in bool_fields:
        setattr(args, name, str(getattr(args, name, "False")).lower() == "true")

    args.data_root = str(project_root / args0.data_root)
    args.cache_dir = str(project_root / args0.cache_dir)
    args.out_dir = str(project_root / args0.out_dir)
    args.rebuild_cache = False
    args.cuda = bool(args0.cuda)
    args.tasks = args0.tasks
    args.seeds = args0.seeds
    args.k_shots = args0.k_shots
    args.repeats = int(args0.repeats)
    args.query_cap_per_class = int(args0.query_cap_per_class)
    if int(args0.source_epochs) > 0:
        args.epochs_ce = int(args0.source_epochs)
    if int(args0.head_epochs) > 0:
        args.linear_epochs = int(args0.head_epochs)
        args.cos_epochs = int(args0.head_epochs)
        args.maha_epochs = int(args0.head_epochs)
    return args


def choose_hash_disjoint_support_query(
    y: np.ndarray,
    allowed_mask: np.ndarray,
    groups: np.ndarray,
    k: int,
    repeat: int,
    seed: int,
    query_cap_per_class: int,
    group_disjoint: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1009 * repeat + 9176 * k + 777331)
    support_idx: List[int] = []
    query_idx: List[int] = []

    for cls in [0, 1]:
        class_idx = np.where((y == cls) & allowed_mask)[0]
        if class_idx.size < k + 1:
            raise RuntimeError(
                f"Not enough hash-disjoint target records for class {cls}: available={class_idx.size}, k={k}"
            )

        if group_disjoint:
            class_groups = np.unique(groups[class_idx])
            rng.shuffle(class_groups)
            selected_groups: List[object] = []
            selected_samples: List[int] = []
            for group in class_groups:
                group_ids = class_idx[groups[class_idx] == group]
                if group_ids.size == 0:
                    continue
                selected_groups.append(group)
                selected_samples.append(int(rng.choice(group_ids)))
                if len(selected_samples) >= k:
                    break
            if len(selected_samples) < k:
                raise RuntimeError(
                    f"Not enough hash/group-disjoint support groups for class {cls}: "
                    f"groups={len(class_groups)}, k={k}"
                )
            support = np.asarray(selected_samples[:k], dtype=np.int64)
            selected_groups_arr = np.asarray(selected_groups, dtype=object)
            query = class_idx[~np.isin(groups[class_idx], selected_groups_arr)]
        else:
            perm = rng.permutation(class_idx)
            support = perm[:k]
            query = perm[k:]

        if query.size == 0:
            raise RuntimeError(f"No query samples remain for class {cls}")
        if query_cap_per_class and query_cap_per_class > 0 and query.size > query_cap_per_class:
            query = rng.choice(query, size=query_cap_per_class, replace=False)

        support_idx.extend([int(i) for i in support])
        query_idx.extend([int(i) for i in query])

    rng.shuffle(support_idx)
    rng.shuffle(query_idx)
    return np.asarray(support_idx, dtype=np.int64), np.asarray(query_idx, dtype=np.int64)


def make_row(lineq, base_info, method, y_query, score_query, support_other_score, use_tc_for_acc_f1, args):
    return lineq.make_row(
        base_info=base_info,
        method=method,
        y_query=y_query,
        score_query=score_query,
        support_other_score=support_other_score,
        use_tc_for_acc_f1=use_tc_for_acc_f1,
        args=args,
    )


def evaluate_episode(
    lineq,
    z_target: np.ndarray,
    y_target: np.ndarray,
    records: pd.DataFrame,
    allowed_mask: np.ndarray,
    task: str,
    k: int,
    seed: int,
    repeat: int,
    args,
    device: torch.device,
    group_disjoint: bool,
    extra_info: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    groups = records["group_id"].to_numpy(dtype=object)
    sup_idx, query_idx = choose_hash_disjoint_support_query(
        y=y_target,
        allowed_mask=allowed_mask,
        groups=groups,
        k=k,
        repeat=repeat,
        seed=seed,
        query_cap_per_class=int(args.query_cap_per_class),
        group_disjoint=group_disjoint,
    )

    z_sup = z_target[sup_idx]
    y_sup = y_target[sup_idx]
    z_q = z_target[query_idx]
    y_q = y_target[query_idx]

    available0 = int(((y_target == 0) & allowed_mask).sum())
    available1 = int(((y_target == 1) & allowed_mask).sum())
    excluded0 = int(((y_target == 0) & (~allowed_mask)).sum())
    excluded1 = int(((y_target == 1) & (~allowed_mask)).sum())

    base_info = {
        "task": task,
        "k_shot": k,
        "seed": seed,
        "repeat": repeat,
        "feature_mode": args.feature_mode,
        "src_tail_weight": args.src_tail_weight,
        "hash_disjoint": True,
        "group_disjoint": bool(group_disjoint),
        "target_available_after_hash_class0": available0,
        "target_available_after_hash_class1": available1,
        "target_excluded_hash_overlap_class0": excluded0,
        "target_excluded_hash_overlap_class1": excluded1,
        "support_count": int(len(sup_idx)),
        "query_count": int(len(query_idx)),
        "query_count_class0": int((y_q == 0).sum()),
        "query_count_class1": int((y_q == 1).sum()),
        "support_groups": int(records.iloc[sup_idx]["group_id"].nunique()),
        "query_groups": int(records.iloc[query_idx]["group_id"].nunique()),
    }
    base_info.update(extra_info)

    rows: List[Dict[str, object]] = []

    anchors = []
    for cls in [0, 1]:
        anchors.append(z_sup[y_sup == cls].mean(axis=0))
    anchors_np = np.stack(anchors, axis=0).astype(np.float32)
    score_q_proto = lineq.cosine_score_np(z_q, anchors_np)
    rows.append(make_row(lineq, base_info, "CE-Proto", y_q, score_q_proto, None, False, args))

    score_sup_lin, score_q_lin = lineq.train_linear_head(z_sup, y_sup, z_q, args, device)
    rows.append(make_row(lineq, base_info, "CE-LinearHead", y_q, score_q_lin, None, False, args))

    score_sup_cos, score_q_cos = lineq.train_cos_head(z_sup, y_sup, z_q, args, device, use_tail=False)
    rows.append(make_row(lineq, base_info, "CosMetricHead", y_q, score_q_cos, None, False, args))

    score_sup_maha, score_q_maha = lineq.train_reg_maha_proxy_head(
        z_sup, y_sup, z_q, args, device, use_tail=False
    )
    rows.append(make_row(
        lineq,
        base_info,
        "RegMahaProxy-TC",
        y_q,
        score_q_maha,
        score_sup_maha[y_sup == 0],
        True,
        args,
    ))

    episode_record = dict(base_info)
    episode_record["support_indices"] = " ".join(map(str, sup_idx.tolist()))
    episode_record["query_indices"] = " ".join(map(str, query_idx.tolist()))
    episode_record["support_hash_unique"] = int(records.iloc[sup_idx]["sha256"].nunique())
    episode_record["query_hash_unique"] = int(records.iloc[query_idx]["sha256"].nunique())
    return rows, episode_record


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iters: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or iters <= 0:
        return float(values.mean()), float(values.mean())
    means = np.empty(int(iters), dtype=float)
    n = values.size
    for i in range(int(iters)):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def holm_adjust(pvals: List[float]) -> List[float]:
    out = [float("nan")] * len(pvals)
    finite = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    running = 0.0
    for rank, (idx, p) in enumerate(finite):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


def summarize(df: pd.DataFrame, episodes: pd.DataFrame, out_dir: Path, bootstrap_iters: int) -> None:
    metrics = ["acc", "f1", "auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01", "pd_tc", "pfa_tc"]
    metric_cols = [m for m in metrics if m in df.columns]

    grouped = df.groupby(["task", "k_shot", "method"])[metric_cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(x) for x in grouped.columns]
    grouped.reset_index().to_csv(out_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    overall = df.groupby("method")[metric_cols].agg(["mean", "std", "count"])
    overall.columns = ["_".join(x) for x in overall.columns]
    overall.reset_index().to_csv(out_dir / "overall_mean_std.csv", index=False, encoding="utf-8-sig")

    keys = ["task", "k_shot", "seed", "repeat"]
    final = "RegMahaProxy-TC"
    baselines = ["CE-Proto", "CosMetricHead", "CE-LinearHead"]
    compare_metrics = ["auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01", "f1"]

    delta_rows: List[Dict[str, object]] = []
    ci_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(20260611)

    for baseline in baselines:
        for metric in compare_metrics:
            a = df[df["method"] == final][keys + [metric]].rename(columns={metric: "final"})
            b = df[df["method"] == baseline][keys + [metric]].rename(columns={metric: "baseline"})
            merged = a.merge(b, on=keys, how="inner")
            merged["delta"] = merged["final"] - merged["baseline"]
            for row in merged.to_dict("records"):
                delta_rows.append({
                    **{k: row[k] for k in keys},
                    "baseline": baseline,
                    "final_method": final,
                    "metric": metric,
                    "baseline_value": row["baseline"],
                    "final_value": row["final"],
                    "delta": row["delta"],
                })
            diff = merged["delta"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(diff, rng=rng, iters=bootstrap_iters)
            ci_rows.append({
                "baseline": baseline,
                "final_method": final,
                "metric": metric,
                "n_pairs": int(np.isfinite(diff).sum()),
                "mean_delta": float(np.nanmean(diff)) if diff.size else float("nan"),
                "median_delta": float(np.nanmedian(diff)) if diff.size else float("nan"),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_ratio": float(np.nanmean(diff > 0)) if diff.size else float("nan"),
                "wilcoxon_p": wilcoxon_p(diff),
            })

    pvals = [float(r["wilcoxon_p"]) for r in ci_rows]
    adj = holm_adjust(pvals)
    for row, p_adj in zip(ci_rows, adj):
        row["holm_p"] = p_adj

    delta = pd.DataFrame(delta_rows)
    delta.to_csv(out_dir / "paired_delta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(out_dir / "paired_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    by_k = delta.groupby(["k_shot", "baseline", "metric"])["delta"].agg(["mean", "std", "count", "median"]).reset_index()
    by_k.to_csv(out_dir / "delta_by_k.csv", index=False, encoding="utf-8-sig")

    lines: List[str] = []
    lines.append("Hash-disjoint robustness summary")
    lines.append("=" * 34)
    lines.append("")
    if not episodes.empty:
        episode_cols = [
            "task",
            "hash_drop_side",
            "source_initial_count",
            "source_after_hash_count",
            "target_available_after_hash_class0",
            "target_available_after_hash_class1",
            "target_excluded_hash_overlap_class0",
            "target_excluded_hash_overlap_class1",
            "query_count_class0",
            "query_count_class1",
        ]
        ep_summary = episodes[episode_cols].drop_duplicates(
            ["task", "hash_drop_side", "target_available_after_hash_class0", "target_available_after_hash_class1"]
        )
        lines.append("Hash filtering strategy and retained records:")
        for _, row in ep_summary.iterrows():
            lines.append(
                "- {task}: drop_side={hash_drop_side}; source records {source_initial_count}->{source_after_hash_count}; "
                "target available other={target_available_after_hash_class0}, target={target_available_after_hash_class1}; "
                "target excluded other={target_excluded_hash_overlap_class0}, target={target_excluded_hash_overlap_class1}".format(**row.to_dict())
            )
        lines.append("")
    lines.append("Overall method means:")
    overall_mean = df.groupby("method")[["auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01", "f1"]].mean().reset_index()
    lines.append(overall_mean.to_string(index=False))
    lines.append("")
    lines.append("Paired RegMahaProxy-TC deltas:")
    lines.append(pd.DataFrame(ci_rows).to_string(index=False))
    (out_dir / "hash_disjoint_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_config(args0: argparse.Namespace, args, out_dir: Path) -> None:
    with (out_dir / "run_config.txt").open("w", encoding="utf-8") as f:
        f.write("[script_args]\n")
        for k, v in sorted(vars(args0).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n[effective_formal_args]\n")
        for k in sorted(vars(args)):
            f.write(f"{k}: {getattr(args, k)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=r"D:\pycharm\py project\uw_metric_code_v3")
    ap.add_argument("--data_root", type=str, default="data_targets")
    ap.add_argument("--tasks", type=str, default="open+speedboat:uuv,open+uuv:speedboat")
    ap.add_argument("--seeds", type=str, default="2026,2027,2028")
    ap.add_argument("--k_shots", type=str, default="5,10,20")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--query_cap_per_class", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="outputs_hash_disjoint_robustness")
    ap.add_argument("--cache_dir", type=str, default="cache_logmel_global_pauc")
    ap.add_argument("--formal_out_dir", type=str, default="outputs_lineq_proxy_formal_b16")
    ap.add_argument("--manifest", type=str, default="outputs_audit/data_targets_manifest.csv")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--source_epochs", type=int, default=-1, help="Use -1 to keep formal source epochs.")
    ap.add_argument("--head_epochs", type=int, default=-1, help="Use -1 to keep formal head epochs.")
    ap.add_argument("--group_disjoint", action="store_true", help="Also enforce inferred support/query group disjointness.")
    ap.add_argument(
        "--drop_side",
        type=str,
        default="auto",
        choices=["auto", "target", "source"],
        help=(
            "target: drop target records whose hash appears in source domains; "
            "source: drop source-training records whose hash appears in the target domain; "
            "auto: use target-side dropping when all requested K values remain feasible, otherwise source-side dropping."
        ),
    )
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    args0 = ap.parse_args()

    project_root = Path(args0.project_root)
    lineq = load_lineq_module(project_root)
    args = load_formal_args(lineq, project_root, args0)

    out_dir = project_root / args0.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), out_dir / "run_hash_disjoint_robustness.py")
    save_config(args0, args, out_dir)

    manifest_path = Path(args0.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_by_rel = read_manifest(manifest_path)

    device = torch.device("cuda" if args0.cuda and torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args0.cuda and device.type != "cuda":
        print("[WARN] --cuda was requested, but CUDA is not available to PyTorch.")

    random.seed(20260611)
    np.random.seed(20260611)

    all_rows: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []
    feasibility_rows: List[Dict[str, object]] = []
    data_root = Path(args.data_root)
    tasks = parse_tasks(args0.tasks)
    seeds = parse_ints(args0.seeds)
    k_shots = parse_ints(args0.k_shots)

    for src_domains, tgt_domain in tasks:
        tname = task_name(src_domains, tgt_domain)
        print("\n" + "=" * 80)
        print(f"TASK {tname}")
        print("=" * 80)

        X_tgt, y_tgt = lineq.load_domain_features(data_root, tgt_domain, args)
        target_records = file_records_for_domain(lineq, data_root, tgt_domain, args, manifest_by_rel)
        if len(target_records) != len(y_tgt) or not np.array_equal(target_records["label"].to_numpy(dtype=np.int64), y_tgt):
            raise RuntimeError(
                f"Target feature/record mismatch for {tgt_domain}: records={len(target_records)}, features={len(y_tgt)}"
            )
        target_hashes = set(target_records["sha256"].astype(str).tolist())

        source_Xs_raw: List[np.ndarray] = []
        source_ys_raw: List[np.ndarray] = []
        source_recs_raw: List[pd.DataFrame] = []
        for domain in src_domains:
            Xd, yd = lineq.load_domain_features(data_root, domain, args)
            recs = file_records_for_domain(lineq, data_root, domain, args, manifest_by_rel)
            if len(recs) != len(yd) or not np.array_equal(recs["label"].to_numpy(dtype=np.int64), yd):
                raise RuntimeError(
                    f"Source feature/record mismatch for {domain}: records={len(recs)}, features={len(yd)}"
                )
            source_Xs_raw.append(Xd)
            source_ys_raw.append(yd)
            source_recs_raw.append(recs)

        raw_source_hashes = set()
        raw_source_count = 0
        for recs in source_recs_raw:
            raw_source_hashes.update(recs["sha256"].astype(str).tolist())
            raw_source_count += int(len(recs))

        target_drop_allowed = ~target_records["sha256"].astype(str).isin(raw_source_hashes).to_numpy()
        feasible_target_drop = all(
            int(((y_tgt == 0) & target_drop_allowed).sum()) >= k + 1
            and int(((y_tgt == 1) & target_drop_allowed).sum()) >= k + 1
            for k in k_shots
        )
        if args0.drop_side == "auto":
            drop_used = "target" if feasible_target_drop else "source"
        else:
            drop_used = args0.drop_side

        if drop_used == "source":
            source_Xs: List[np.ndarray] = []
            source_ys: List[np.ndarray] = []
            source_recs: List[pd.DataFrame] = []
            for Xd, yd, recs in zip(source_Xs_raw, source_ys_raw, source_recs_raw):
                keep = ~recs["sha256"].astype(str).isin(target_hashes).to_numpy()
                source_Xs.append(Xd[keep])
                source_ys.append(yd[keep])
                source_recs.append(recs.loc[keep].reset_index(drop=True))
            source_hashes = set()
            for recs in source_recs:
                source_hashes.update(recs["sha256"].astype(str).tolist())
            allowed_mask = np.ones(len(target_records), dtype=bool)
        else:
            source_Xs = source_Xs_raw
            source_ys = source_ys_raw
            source_hashes = raw_source_hashes
            allowed_mask = target_drop_allowed

        X_src = np.concatenate(source_Xs, axis=0)
        y_src = np.concatenate(source_ys, axis=0)
        source_after_count = int(len(y_src))

        print(
            f"[HASH] drop_side={drop_used}, raw source records={raw_source_count}, "
            f"source after hash filter={source_after_count}, source unique hashes={len(source_hashes)}, "
            f"target excluded={int((~allowed_mask).sum())}/{len(allowed_mask)}"
        )
        print(
            f"[TARGET AFTER HASH] other={int(((y_tgt == 0) & allowed_mask).sum())}, "
            f"target={int(((y_tgt == 1) & allowed_mask).sum())}"
        )

        if args.feature_norm == "global":
            X_src, X_tgt = lineq.apply_global_feature_norm(X_src, X_tgt)
        X_src = lineq.maybe_replicate_single_to_rgb(X_src, args)
        X_tgt = lineq.maybe_replicate_single_to_rgb(X_tgt, args)

        print(f"[SOURCE] X={X_src.shape}, other={int((y_src == 0).sum())}, target={int((y_src == 1).sum())}")
        print(f"[TARGET] X={X_tgt.shape}, other={int((y_tgt == 0).sum())}, target={int((y_tgt == 1).sum())}")

        feasible_k = []
        for k in k_shots:
            ok = (
                int(((y_tgt == 0) & allowed_mask).sum()) >= k + 1
                and int(((y_tgt == 1) & allowed_mask).sum()) >= k + 1
            )
            feasible_k.append(k if ok else None)
            feasibility_rows.append({
                "task": tname,
                "k_shot": k,
                "drop_side_requested": args0.drop_side,
                "drop_side_used": drop_used,
                "feasible": bool(ok),
                "target_available_after_hash_class0": int(((y_tgt == 0) & allowed_mask).sum()),
                "target_available_after_hash_class1": int(((y_tgt == 1) & allowed_mask).sum()),
                "target_excluded_hash_overlap_class0": int(((y_tgt == 0) & (~allowed_mask)).sum()),
                "target_excluded_hash_overlap_class1": int(((y_tgt == 1) & (~allowed_mask)).sum()),
                "source_initial_count": raw_source_count,
                "source_after_hash_count": source_after_count,
            })
        pd.DataFrame(feasibility_rows).to_csv(out_dir / "hash_disjoint_feasibility.csv", index=False, encoding="utf-8-sig")
        if not any(x is not None for x in feasible_k):
            print(f"[SKIP] no feasible K-shot episode remains for task={tname} with drop_side={drop_used}")
            continue

        extra_info = {
            "hash_drop_side": drop_used,
            "hash_drop_side_requested": args0.drop_side,
            "source_initial_count": raw_source_count,
            "source_after_hash_count": source_after_count,
            "target_source_hash_overlap_raw": int((~target_drop_allowed).sum()),
        }

        for seed in seeds:
            print(f"\n[SEED] {seed} train source encoder")
            encoder, _ = lineq.train_source_encoder(X_src, y_src, args, device, seed=seed)
            z_tgt = lineq.encode_array(encoder, X_tgt, device=device, batch_size=max(64, int(args.batch_size) * 8))

            for k in k_shots:
                if not (
                    int(((y_tgt == 0) & allowed_mask).sum()) >= k + 1
                    and int(((y_tgt == 1) & allowed_mask).sum()) >= k + 1
                ):
                    print(f"[SKIP] task={tname} k={k} infeasible after hash filtering")
                    continue
                for rep in range(int(args0.repeats)):
                    rows, ep = evaluate_episode(
                        lineq=lineq,
                        z_target=z_tgt,
                        y_target=y_tgt,
                        records=target_records,
                        allowed_mask=allowed_mask,
                        task=tname,
                        k=k,
                        seed=seed,
                        repeat=rep,
                        args=args,
                        device=device,
                        group_disjoint=bool(args0.group_disjoint),
                        extra_info=extra_info,
                    )
                    all_rows.extend(rows)
                    episode_rows.append(ep)
                print(f"done task={tname} seed={seed} k={k}")
                pd.DataFrame(all_rows).to_csv(out_dir / "all_metrics_partial.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(episode_rows).to_csv(out_dir / "manifest_used_by_episode.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(all_rows)
    episodes = pd.DataFrame(episode_rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(out_dir / "manifest_used_by_episode.csv", index=False, encoding="utf-8-sig")
    summarize(df, episodes, out_dir, bootstrap_iters=int(args0.bootstrap_iters))
    print(f"Wrote {out_dir / 'all_metrics.csv'}")
    print(f"Wrote {out_dir / 'paired_bootstrap_ci.csv'}")
    print(f"Wrote {out_dir / 'hash_disjoint_summary.txt'}")


if __name__ == "__main__":
    main()
