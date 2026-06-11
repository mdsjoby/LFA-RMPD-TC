# -*- coding: utf-8 -*-
"""
Group-disjoint robustness check for the LFA-RMPD-TC manuscript.

This script reuses the formal log-mel feature cache and the method
implementations in run_lineq_proxy_pauc_cmtc.py, but changes target-domain
episode construction so that support and query WAV records come from disjoint
inferred recording groups.

The inferred group id is obtained from the QiandaoEar22 filename prefix before
the trailing "_label__<class>__<segment>" suffix.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_tasks(s: str) -> List[Tuple[List[str], str]]:
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        src, tgt = item.split(":", 1)
        out.append(([x.strip() for x in src.split("+") if x.strip()], tgt.strip()))
    return out


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def infer_group_id(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_label__\d+__\d+(?: - 副本)?$", "", stem)
    stem = re.sub(r"_label__\d+__\d+(?: - copy)?$", "", stem, flags=re.IGNORECASE)
    return stem


def file_groups_for_domain(lineq, data_root: Path, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    label_files = lineq.find_label_files(data_root / domain)
    files = label_files[0] + label_files[1]
    y = np.asarray([0] * len(label_files[0]) + [1] * len(label_files[1]), dtype=np.int64)
    groups = np.asarray([infer_group_id(p) for p in files], dtype=object)
    return groups, y


def choose_group_disjoint_support_query(
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
    repeat: int,
    seed: int,
    query_cap_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1009 * repeat + 9176 * k + 424242)
    support_idx: List[int] = []
    query_idx: List[int] = []

    for c in [0, 1]:
        class_idx = np.where(y == c)[0]
        if class_idx.size == 0:
            raise RuntimeError(f"No samples for class {c}")
        class_groups = np.unique(groups[class_idx])
        rng.shuffle(class_groups)

        selected_groups: List[object] = []
        selected_samples: List[int] = []
        for g in class_groups:
            ids = class_idx[groups[class_idx] == g]
            if ids.size == 0:
                continue
            selected_groups.append(g)
            selected_samples.append(int(rng.choice(ids)))
            if len(selected_samples) >= k:
                break

        if len(selected_samples) < k:
            # Extremely small class/group fallback; keep group-disjoint query.
            selected_samples = list(rng.choice(class_idx, size=k, replace=True))
            selected_groups = list(np.unique(groups[selected_samples]))

        selected_groups_arr = np.asarray(selected_groups, dtype=object)
        support_idx.extend(selected_samples[:k])

        q = class_idx[~np.isin(groups[class_idx], selected_groups_arr)]
        if q.size == 0:
            raise RuntimeError(f"No group-disjoint query samples for class {c}")
        if query_cap_per_class and query_cap_per_class > 0 and q.size > query_cap_per_class:
            q = rng.choice(q, size=query_cap_per_class, replace=False)
        query_idx.extend([int(i) for i in q])

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


def evaluate_group_episode(lineq, z_target, y_target, groups, task, k, seed, repeat, args, device):
    sup_idx, query_idx = choose_group_disjoint_support_query(
        y=y_target,
        groups=groups,
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
        "group_disjoint": True,
        "support_groups": int(len(np.unique(groups[sup_idx]))),
        "query_groups": int(len(np.unique(groups[query_idx]))),
    }

    rows = []

    anchors = []
    for c in [0, 1]:
        anchors.append(z_sup[y_sup == c].mean(axis=0))
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
    return rows


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["acc", "f1", "auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01", "pd_tc", "pfa_tc"]
    metric_cols = [m for m in metrics if m in df.columns]
    summary = df.groupby(["task", "k_shot", "method"])[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    overall = df.groupby("method")[metric_cols].agg(["mean", "std", "count"])
    overall.columns = ["_".join(x) for x in overall.columns]
    overall = overall.reset_index()
    overall.to_csv(out_dir / "overall_mean_std.csv", index=False, encoding="utf-8-sig")

    rows = []
    final = "RegMahaProxy-TC"
    keys = ["task", "k_shot", "seed", "repeat"]
    for baseline in ["CE-Proto", "CosMetricHead", "CE-LinearHead"]:
        for metric in ["auc", "pd_at_pfa_0.05", "pd_at_pfa_0.01", "f1"]:
            a = df[df.method == final][keys + [metric]].rename(columns={metric: "final"})
            b = df[df.method == baseline][keys + [metric]].rename(columns={metric: "baseline"})
            m = a.merge(b, on=keys, how="inner")
            diff = m["final"] - m["baseline"]
            rows.append({
                "baseline": baseline,
                "metric": metric,
                "n_pairs": int(diff.notna().sum()),
                "mean_delta": float(diff.mean()),
                "median_delta": float(diff.median()),
                "positive_ratio": float((diff > 0).mean()),
            })
    pd.DataFrame(rows).to_csv(out_dir / "paired_delta_summary.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=r"D:\pycharm\py project\uw_metric_code_v3")
    ap.add_argument("--data_root", type=str, default="data_targets")
    ap.add_argument("--tasks", type=str, default="open+speedboat:uuv,open+uuv:speedboat")
    ap.add_argument("--seeds", type=str, default="2026,2027,2028")
    ap.add_argument("--k_shots", type=str, default="5,10,20")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--query_cap_per_class", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default="outputs_group_disjoint_robustness")
    ap.add_argument("--cache_dir", type=str, default="cache_logmel_global_pauc")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--fast_epochs", type=int, default=10)
    ap.add_argument(
        "--source_epochs",
        type=int,
        default=-1,
        help="Override source encoder epochs. Use -1 to keep the formal run_config value.",
    )
    args0 = ap.parse_args()

    project_root = Path(args0.project_root)
    lineq = load_lineq_module(project_root)
    lineq_args = lineq.parse_args([
        "--tasks", args0.tasks,
        "--data_root", str(project_root / args0.data_root),
        "--out_dir", str(project_root / args0.out_dir),
    ]) if False else None

    # Use the formal run configuration, with reduced head epochs for robustness.
    class Args:
        pass
    args = Args()
    formal_cfg = project_root / "outputs_lineq_proxy_formal_b16" / "run_config.txt"
    cfg: Dict[str, str] = {}
    for line in formal_cfg.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            cfg[k] = v
    for k, v in cfg.items():
        setattr(args, k, v)
    # Cast required fields.
    int_fields = [
        "target_sr", "n_fft", "hop_length", "win_length", "feature_time_bins",
        "stft_freq_bins", "mel_n_mels", "batch_size", "emb_dim", "encoder_width",
        "epochs_ce", "linear_epochs", "cos_epochs", "maha_epochs", "src_pauc_warmup_epochs",
        "hpss_harmonic_kernel", "hpss_percussive_kernel",
    ]
    float_fields = [
        "segment_sec", "segment_hop_sec", "mel_fmin", "mel_fmax", "dropout",
        "lr_ce", "weight_decay", "lr_linear", "lr_cos", "cos_scale",
        "lr_maha", "maha_scale", "maha_proxy_l2", "maha_diag_l2",
        "maha_bias_l2", "cos_tail_weight", "cos_tail_margin",
        "cos_tail_quantile", "src_tail_weight", "src_pauc_weight",
        "src_pauc_alpha", "src_pauc_beta", "src_pauc_margin",
    ]
    for f in int_fields:
        setattr(args, f, int(getattr(args, f)))
    for f in float_fields:
        setattr(args, f, float(getattr(args, f)))
    for f in ["single_to_rgb", "maha_fixed_scale", "calib_cos_fixed_scale", "na_calib_cos_fixed_scale", "verbose"]:
        setattr(args, f, str(getattr(args, f, "False")).lower() == "true")
    args.data_root = str(project_root / args0.data_root)
    args.cache_dir = str(project_root / args0.cache_dir)
    args.out_dir = str(project_root / args0.out_dir)
    args.rebuild_cache = False
    args.cuda = args0.cuda
    args.tasks = args0.tasks
    args.seeds = args0.seeds
    args.k_shots = args0.k_shots
    args.repeats = args0.repeats
    args.query_cap_per_class = args0.query_cap_per_class
    if args0.source_epochs > 0:
        args.epochs_ce = args0.source_epochs
    args.linear_epochs = args0.fast_epochs
    args.cos_epochs = args0.fast_epochs
    args.maha_epochs = args0.fast_epochs

    out_dir = project_root / args0.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), out_dir / "run_group_disjoint_robustness.py")
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        for k in sorted(vars(args)):
            f.write(f"{k}: {getattr(args, k)}\n")

    device = torch.device("cuda" if args0.cuda and torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    all_rows = []
    data_root = Path(args.data_root)
    tasks = parse_tasks(args0.tasks)
    seeds = parse_ints(args0.seeds)
    k_shots = parse_ints(args0.k_shots)

    for src_domains, tgt_domain in tasks:
        tname = "+".join(src_domains) + "->" + tgt_domain
        print("=" * 80)
        print(tname)
        source_Xs, source_ys = [], []
        for d in src_domains:
            Xd, yd = lineq.load_domain_features(data_root, d, args)
            source_Xs.append(Xd)
            source_ys.append(yd)
        X_src = np.concatenate(source_Xs, axis=0)
        y_src = np.concatenate(source_ys, axis=0)
        X_tgt, y_tgt = lineq.load_domain_features(data_root, tgt_domain, args)
        groups, y_from_files = file_groups_for_domain(lineq, data_root, tgt_domain)
        if len(groups) != len(y_tgt) or not np.array_equal(y_from_files, y_tgt):
            raise RuntimeError(f"Group metadata mismatch for target={tgt_domain}")

        if args.feature_norm == "global":
            X_src, X_tgt = lineq.apply_global_feature_norm(X_src, X_tgt)
        X_src = lineq.maybe_replicate_single_to_rgb(X_src, args)
        X_tgt = lineq.maybe_replicate_single_to_rgb(X_tgt, args)

        for seed in seeds:
            print(f"seed={seed} train source encoder")
            encoder, _ = lineq.train_source_encoder(X_src, y_src, args, device, seed=seed)
            z_tgt = lineq.encode_array(encoder, X_tgt, device=device, batch_size=max(64, args.batch_size * 8))

            for k in k_shots:
                for rep in range(args0.repeats):
                    rows = evaluate_group_episode(lineq, z_tgt, y_tgt, groups, tname, k, seed, rep, args, device)
                    all_rows.extend(rows)
                print(f"done task={tname} seed={seed} k={k}")
                pd.DataFrame(all_rows).to_csv(out_dir / "all_metrics_partial.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    summarize(df, out_dir)
    print(out_dir / "all_metrics.csv")


if __name__ == "__main__":
    main()
