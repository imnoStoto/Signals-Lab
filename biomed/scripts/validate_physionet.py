#!/usr/bin/env python3
"""
validate_physionet.py

Validates a PhysioNet/WFDB dataset using the ingestion manifest.

Inputs:
- manifest.csv or manifest.parquet created by ingest_physionet.py

Outputs (written to --out):
- record_quality.csv
- record_quality.parquet (optional if parquet engine available)
- usable_records.csv
- validation_summary.json

What validation means here:
- data engineering / pipeline sanity checks (NOT clinical inference)
- structural integrity (metadata reasonable)
- file presence (.hea/.dat exist)
- optional sample sanity checks (finite values, non-flatline) if:
    a) preview fields exist in manifest, OR
    b) --read-samples-if-missing is enabled

This script does NOT use wfdb rd_dir; it uses record_base_path strings.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import wfdb
except ImportError as e:
    raise SystemExit(
        "Missing dependency: wfdb.\n"
        "Install it in your venv with: pip install wfdb\n"
        f"Original error: {e}"
    )


# -----------------------------
# Validation thresholds (edit later if you want)
# -----------------------------

@dataclass(frozen=True)
class Thresholds:
    min_fs: float = 1.0                 # sampling rate must be > 0
    min_sig_len: int = 1                # must have at least 1 sample
    min_duration_sec: float = 10.0      # ignore tiny/empty records
    max_duration_sec: float = 24 * 3600 # sanity: <= 24 hours
    flatline_eps: float = 1e-12         # mean absolute deviation threshold for "flat"


# -----------------------------
# Helpers
# -----------------------------

def load_manifest(path: Path) -> pd.DataFrame:
    """
    Load manifest from CSV or Parquet.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("Manifest must be .csv or .parquet")


def safe_bool(x) -> bool:
    return bool(x) if pd.notna(x) else False


def check_files(record_base_path: str) -> tuple[bool, bool]:
    """
    Check presence of .hea and .dat for a given record base path string.
    """
    base = Path(record_base_path)
    return base.with_suffix(".hea").exists(), base.with_suffix(".dat").exists()


def compute_sample_sanity(record_base_path: str, fs: float, max_seconds: float) -> dict:
    """
    Read first N seconds of samples and compute basic sanity checks.
    Returns:
      - sample_finite_ok (bool)
      - sample_flat_ok (bool)
      - sample_preview_samples (int)
      - sample_preview_seconds (float)
      - sample_min/max/mean (float)
      - sample_mad (float) mean abs deviation
    """
    import numpy as np

    base = str(Path(record_base_path))
    preview_samples = int(fs * max_seconds) if fs > 0 else 0
    if preview_samples <= 0:
        return {"sample_check_ran": False}

    signals, _fields = wfdb.rdsamp(base, sampfrom=0, sampto=preview_samples)

    flat = signals.astype("float64").ravel()
    if flat.size == 0:
        return {"sample_check_ran": False}

    finite_ok = bool(np.isfinite(flat).all())
    mean = float(np.mean(flat))
    mad = float(np.mean(np.abs(flat - mean)))  # robust-ish flatness proxy

    return {
        "sample_check_ran": True,
        "sample_preview_samples": int(preview_samples),
        "sample_preview_seconds": float(preview_samples / fs) if fs > 0 else None,
        "sample_min": float(np.min(flat)),
        "sample_max": float(np.max(flat)),
        "sample_mean": mean,
        "sample_mad": mad,
        "sample_finite_ok": finite_ok,
        "sample_flat_ok": bool(mad > Thresholds().flatline_eps),
    }


def add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


# -----------------------------
# Main validation logic
# -----------------------------

def validate_row(row: pd.Series, t: Thresholds, read_samples_if_missing: bool, max_seconds: float) -> dict:
    """
    Validate one record row from manifest.
    Returns a dict of flags + reasons + usable boolean.
    """
    reasons: list[str] = []

    record_id = str(row.get("record_id", row.get("record_name", "")))
    base_path = row.get("record_base_path", None)

    if not base_path or (isinstance(base_path, float) and pd.isna(base_path)):
        add_reason(reasons, "missing_record_base_path")
        # Without base path, we can't do file checks or sample checks.
        return {
            "record_id": record_id,
            "usable": False,
            "reasons": ";".join(reasons),
            "hea_exists": False,
            "dat_exists": False,
        }

    # ---- Structural checks from header fields (manifest)
    fs = float(row.get("fs", 0) or 0)
    sig_len = int(row.get("sig_len", 0) or 0)
    n_sig = int(row.get("n_sig", 0) or 0)
    duration_sec = float(row.get("duration_sec", 0) or 0)

    if fs < t.min_fs:
        add_reason(reasons, "invalid_fs")
    if sig_len < t.min_sig_len:
        add_reason(reasons, "invalid_sig_len")
    if n_sig < 1:
        add_reason(reasons, "invalid_n_sig")
    if duration_sec < t.min_duration_sec:
        add_reason(reasons, "too_short")
    if duration_sec > t.max_duration_sec:
        add_reason(reasons, "too_long")

    # ---- File presence checks
    hea_exists, dat_exists = check_files(str(base_path))
    if not hea_exists:
        add_reason(reasons, "missing_hea")
    if not dat_exists:
        add_reason(reasons, "missing_dat")

    # ---- Sample checks
    # Prefer preview stats from ingest, if present.
    preview_mad = None
    sample_check = {"sample_check_ran": False}

    # If ingest computed preview stats, use them
    # (We didn’t compute MAD in ingest; we can approximate flatness via min/max, but MAD is better.
    # If MAD isn't present, we can still run a sample check optionally.)
    preview_min = row.get("preview_min", None)
    preview_max = row.get("preview_max", None)

    has_preview = pd.notna(preview_min) and pd.notna(preview_max)

    if has_preview:
        # Quick flatline proxy: min==max means totally flat in preview window.
        if float(preview_max) == float(preview_min):
            add_reason(reasons, "flat_preview")
    else:
        if read_samples_if_missing and hea_exists and dat_exists and fs >= t.min_fs:
            try:
                sample_check = compute_sample_sanity(str(base_path), fs=fs, max_seconds=max_seconds)
                if sample_check.get("sample_check_ran"):
                    if not sample_check.get("sample_finite_ok", True):
                        add_reason(reasons, "nonfinite_samples")
                    if not sample_check.get("sample_flat_ok", True):
                        add_reason(reasons, "flat_samples")
                    preview_mad = sample_check.get("sample_mad", None)
            except Exception:
                add_reason(reasons, "sample_check_failed")

    # ---- Final usable decision
    usable = (len(reasons) == 0)

    out = {
        "record_id": record_id,
        "record_base_path": str(base_path),
        "fs": fs,
        "sig_len": sig_len,
        "n_sig": n_sig,
        "duration_sec": duration_sec,
        "hea_exists": hea_exists,
        "dat_exists": dat_exists,
        "usable": usable,
        "reasons": ";".join(reasons),
    }

    # Add sample_check fields if we ran it
    if sample_check.get("sample_check_ran"):
        out.update(sample_check)
    if preview_mad is not None:
        out["sample_mad"] = preview_mad

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate WFDB dataset using an ingestion manifest.")
    p.add_argument("--manifest", required=True, help="Path to manifest.csv or manifest.parquet")
    p.add_argument("--out", required=True, help="Output directory for validation artifacts")
    p.add_argument("--read-samples-if-missing", action="store_true",
                   help="If manifest lacks preview stats, read first N seconds for sanity checks.")
    p.add_argument("--max-seconds", type=float, default=10.0,
                   help="Seconds to read per record if sample sanity checks run.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def try_write_parquet(df: pd.DataFrame, path: Path, log: logging.Logger) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        log.warning("Could not write parquet to %s (%s).", path, e)


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("validate")

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return 2

    log.info("Loading manifest: %s", manifest_path)
    df = load_manifest(manifest_path)

    # Basic expectation: must have record_base_path
    if "record_base_path" not in df.columns:
        log.error("Manifest is missing required column: record_base_path")
        log.error("Re-run ingest_physionet.py (fresh version) to generate a compatible manifest.")
        return 3

    t = Thresholds()

    log.info("Validating %d records...", len(df))
    results = []
    for _, row in df.iterrows():
        results.append(
            validate_row(row, t, read_samples_if_missing=args.read_samples_if_missing, max_seconds=args.max_seconds)
        )

    out_df = pd.DataFrame(results)

    # Write outputs
    record_quality_csv = out_dir / "record_quality.csv"
    record_quality_parquet = out_dir / "record_quality.parquet"
    usable_csv = out_dir / "usable_records.csv"
    summary_json = out_dir / "validation_summary.json"

    out_df.to_csv(record_quality_csv, index=False)
    try_write_parquet(out_df, record_quality_parquet, log)

    usable = out_df[out_df["usable"] == True][["record_id", "record_base_path"]].copy()
    usable.to_csv(usable_csv, index=False)

    summary = {
        "total_records": int(len(out_df)),
        "usable_records": int((out_df["usable"] == True).sum()),
        "failed_records": int((out_df["usable"] == False).sum()),
        "failure_reason_counts": (
            out_df.loc[out_df["usable"] == False, "reasons"]
            .str.split(";")
            .explode()
            .value_counts()
            .to_dict()
        ),
        "thresholds": t.__dict__,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Wrote: %s", record_quality_csv)
    log.info("Wrote: %s", usable_csv)
    log.info("Wrote: %s", summary_json)
    log.info("Done. Usable=%d / %d", summary["usable_records"], summary["total_records"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
