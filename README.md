# LFA-RMPD-TC for Few-Shot Underwater Acoustic Target Detection

This repository provides code, metadata, configuration files, result CSV files, and plotting scripts for the manuscript:

**Low-false-alarm-aware metric learning for few-shot underwater acoustic target detection**

## Dataset

The raw audio data are derived from the public QiandaoEar22 dataset:

https://github.com/xiaoyangdu22/QiandaoEar22

This repository does **not** redistribute raw WAV files. Users should download the raw data from the official QiandaoEar22 source and organize the local `data_targets` directory according to the metadata provided here.

Local aliases used in the experiments:

- `open` -> official KaiYuan subdataset
- `speedboat` -> official SpeedBoat subdataset
- `uuv` -> official UUV subdataset

## Repository Contents

- `scripts/`: experiment and plotting scripts.
- `configs/`: run configuration files recorded from the reported experiments.
- `metadata/`: processed domain metadata and official-subdataset mapping.
- `results/main_matched_episode/`: CSV results for the main matched-episode protocol.
- `results/audit/`: record-level SHA256 audit outputs for the local `data_targets` WAV files.
- `results/source_hash_filtered/`: CSV results for the source-hash-filtered robustness check.
- `results/group_disjoint/`: CSV results for the group-disjoint robustness check.
- `figures/`: manuscript figures generated from the reported experiments.

## Main Experiments

The main matched-episode protocol evaluates:

- KaiYuan + SpeedBoat -> UUV
- KaiYuan + UUV -> SpeedBoat

The reported main results correspond to `results/main_matched_episode`.

The SHA256 audit reported in the manuscript corresponds to `results/audit`. It records the local preprocessed WAV records used in the experiments, summarizes their durations and class counts, and reports cross-domain file-hash overlap.

The source-hash-filtered robustness check removes source-training records whose SHA256 hashes appear in the target domain before source encoder training. The reported results correspond to `results/source_hash_filtered`.

The group-disjoint robustness check uses available recording-group identifiers derived from QiandaoEar22 file names and is reported in `results/group_disjoint`.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main matched-episode experiment from a local project directory containing `data_targets`:

```bash
python scripts/run_main_matched_episode.py --data_root data_targets --tasks open+speedboat:uuv,open+uuv:speedboat --seeds 2026,2027,2028 --k_shots 5,10,20,50 --epochs_ce 25 --repeats 10 --query_cap_per_class 500 --batch_size 16 --cuda --out_dir outputs_reproduced_main
```

Run the group-disjoint robustness check:

```bash
python scripts/run_group_disjoint_robustness.py --data_root data_targets --out_dir outputs_reproduced_group_disjoint
```

Run the record-level data audit:

```bash
python scripts/audit_data_targets.py --data_root data_targets --out_dir outputs_audit --domains open,speedboat,uuv --segment_sec 4.0
python scripts/check_background_overlap.py --manifest outputs_audit/data_targets_manifest.csv --out_dir outputs_audit --tasks open+speedboat:uuv,open+uuv:speedboat
```

Run the source-hash-filtered robustness check:

```bash
python scripts/run_hash_disjoint_robustness.py --data_root data_targets --tasks open+speedboat:uuv,open+uuv:speedboat --seeds 2026,2027,2028 --k_shots 5,10,20,50 --repeats 10 --query_cap_per_class 500 --out_dir outputs_hash_disjoint_robustness --cache_dir cache_logmel_global_pauc --cuda --drop_side source --source_epochs -1 --head_epochs 120 --bootstrap_iters 5000
```

Generate acoustic example figures:

```bash
python scripts/plot_qiandaoear22_logmel_examples.py --data_root data_targets --out_dir figures
```

## Notes

The run configuration files in `configs/` are the authoritative record of the reported experimental settings. Depending on GPU, CUDA, package versions, and random number behavior, exact floating-point reproduction may vary slightly.

Large raw audio files, trained checkpoints, temporary outputs, and local environment files are intentionally not included. Please download QiandaoEar22 from the official source and keep the raw data in a local `data_targets/` directory.

## Citation

Citation information will be added after publication.
