#!/usr/bin/env python3
"""
features_ecg_baseline.py

Compute simple baseline features for WFDB ECG records listed in usable_records.csv.

Inputs:
- usable_records.csv with columns:
    record_id, record_base_path

Outputs (written to --out):
- ecg_features.csv
- ecg_features.parquet (optional if parquet engine exists)

Features computed per record (over a time window):
- channel_count
- fs
- window_seconds, window_samples
- per-channel: mean, std, rms, min, max, ptp (peak-to-peak), zero_cross_rate
- crude heart rate estimate (per channel): using peak detection on a filtered signal (optional)

This is designed to be:
- deterministic
- robust across wfdb versions (no rd_dir)
- useful for data-architecture demonstrations (pipeline artifacts)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import wfdb
except ImportError as e:
    raise SystemExit(
        "Missing dependency: wfdb.\n"
        "Install with: pip install wfdb\n"
        f"Original error: {e}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute baseline ECG features from usable WFDB records.")
    p.add_argument("--usable", required=True, help="Path to usable_records.csv")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--seconds", type=float, default=30.0, help="Window length to analyze (seconds)")
    p.add_argument("--start-sec", type=float, default=0.0, help="Start time (seconds) into record")
    p.add_argument("--max-records", type=int, default=0, help="If >0, only process first N records")
    p.add_argument("--channels", type=str, default="", help="Comma-separated channel indices (e.g. '0' or '0,1'). Default: all")
    p.add_argument("--estimate-hr", action="store_true", help="Estimate heart rate via simple peak detection (crude).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def try_write_parquet(df: pd.DataFrame, path: Path, log: logging.Logger) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        log.warning("Could not write parquet to %s (%s).", path, e)


def parse_channels_arg(ch_arg: str) -> list[int] | None:
    ch_arg = (ch_arg or "").strip()
    if not ch_arg:
        return None
    out = []
    for part in ch_arg.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out if out else None


def compute_basic_stats(x) -> dict[str, float]:
    """
    Compute basic numeric stats for a 1D signal x (NumPy array).
    """
    import numpy as np

    x = x.astype("float64")
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    mn = float(np.min(x))
    mx = float(np.max(x))
    ptp = float(mx - mn)

    # Zero-crossing rate (rough signal activity proxy)
    # Count sign changes / length
    s = np.sign(x)
    zc = float(np.mean(s[1:] != s[:-1])) if x.size > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": mn,
        "max": mx,
        "ptp": ptp,
        "zero_cross_rate": zc,
    }


def estimate_hr_bpm(x, fs: float) -> dict[str, Any]:
    """
    Crude HR estimate:
    - bandpass-ish via moving average subtraction (highpass-ish)
    - detect peaks with a dynamic threshold
    This is not clinical-grade. It’s just a pipeline demo feature.

    Returns bpm and peak count.
    """
    import numpy as np

    if fs <= 0 or x.size < int(fs * 5):
        return {"hr_bpm": None, "hr_peaks": 0}

    x = x.astype("float64")

    # Simple detrend: subtract a moving average (removes baseline wander)
    win = int(max(1, fs * 0.75))  # ~0.75s window
    kernel = np.ones(win) / win
    baseline = np.convolve(x, kernel, mode="same")
    y = x - baseline

    # Rectify and smooth to get an "energy envelope"
    env = np.abs(y)
    env = np.convolve(env, np.ones(int(max(1, fs * 0.10))) / int(max(1, fs * 0.10)), mode="same")  # ~100ms

    # Dynamic threshold: median + k * MAD
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    thr = med + 6.0 * mad

    # Peak detection: local maxima above threshold with refractory period
    refractory = int(fs * 0.25)  # 250ms
    peaks = []
    last = -refractory

    for i in range(1, len(env) - 1):
        if i - last < refractory:
            continue
        if env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            peaks.append(i)
            last = i

    duration_sec = x.size / fs
    if duration_sec <= 0:
        return {"hr_bpm": None, "hr_peaks": int(len(peaks))}

    bpm = (len(peaks) / duration_sec) * 60.0
    return {"hr_bpm": float(bpm), "hr_peaks": int(len(peaks))}


def load_window(record_base_path: str, start_sec: float, seconds: float):
    """
    Load a time window from a WFDB record using full record base path (no rd_dir).
    """
    header = wfdb.rdheader(record_base_path)
    fs = float(header.fs)
    sig_len = int(header.sig_len)

    start = int(max(0, start_sec) * fs)
    count = int(max(0.0, seconds) * fs)
    end = min(sig_len, start + count)

    if end <= start:
        return None

    signals, fields = wfdb.rdsamp(record_base_path, sampfrom=start, sampto=end)
    return signals, fields, fs, start, end


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("features")

    usable_path = Path(args.usable).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not usable_path.exists():
        log.error("usable_records.csv not found: %s", usable_path)
        return 2

    df_usable = pd.read_csv(usable_path)

    required = {"record_id", "record_base_path"}
    missing = required - set(df_usable.columns)
    if missing:
        log.error("usable_records.csv missing required columns: %s", sorted(missing))
        return 3

    if args.max_records and args.max_records > 0:
        df_usable = df_usable.head(args.max_records)

    channels = parse_channels_arg(args.channels)

    results: list[dict[str, Any]] = []
    failures = 0

    for _, row in df_usable.iterrows():
        record_id = str(row["record_id"])
        record_base_path = str(row["record_base_path"])

        try:
            loaded = load_window(record_base_path, start_sec=args.start_sec, seconds=args.seconds)
            if loaded is None:
                failures += 1
                log.warning("Empty window: %s", record_id)
                continue

            signals, fields, fs, start, end = loaded
            n_samples, n_sig = signals.shape

            # Choose channels
            ch_list = list(range(n_sig)) if channels is None else [c for c in channels if 0 <= c < n_sig]
            if not ch_list:
                failures += 1
                log.warning("No valid channels selected for %s (n_sig=%d)", record_id, n_sig)
                continue

            base_row = {
                "record_id": record_id,
                "record_base_path": record_base_path,
                "fs": fs,
                "channel_count": int(n_sig),
                "window_start_sample": int(start),
                "window_end_sample": int(end),
                "window_samples": int(n_samples),
                "window_seconds": float(n_samples / fs) if fs > 0 else None,
            }

            for ch in ch_list:
                x = signals[:, ch]

                stats = compute_basic_stats(x)
                feat = dict(base_row)
                feat["channel_index"] = int(ch)

                # Optional: include signal name if available
                sig_names = fields.get("sig_name", None)
                feat["channel_name"] = sig_names[ch] if isinstance(sig_names, list) and ch < len(sig_names) else None

                # Add stats with stable column names
                for k, v in stats.items():
                    feat[f"{k}"] = v

                if args.estimate_hr:
                    feat.update(estimate_hr_bpm(x, fs=fs))

                results.append(feat)

            log.debug("OK: %s", record_id)

        except Exception as e:
            failures += 1
            log.warning("Failed record %s: %s", record_id, e)

    if not results:
        log.error("No features computed. Failures=%d", failures)
        return 4

    out_df = pd.DataFrame(results)

    csv_path = out_dir / "ecg_features.csv"
    pq_path = out_dir / "ecg_features.parquet"
    summary_path = out_dir / "features_summary.json"

    out_df.to_csv(csv_path, index=False)
    try_write_parquet(out_df, pq_path, log)

    summary = {
        "records_requested": int(len(df_usable)),
        "rows_written": int(len(out_df)),
        "failures": int(failures),
        "window_seconds": float(args.seconds),
        "start_sec": float(args.start_sec),
        "estimate_hr": bool(args.estimate_hr),
        "channels": channels if channels is not None else "all",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Wrote: %s", csv_path)
    log.info("Wrote: %s", summary_path)
    log.info("Done. Rows=%d Failures=%d", len(out_df), failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
