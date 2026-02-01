#!/usr/bin/env python3  # Allows running as ./scripts/ingest_physionet.py on Unix if executable (optional)

"""  
A robust ingestion tool for PhysioNet WFDB datasets (e.g., MIT-BIH) that:
1) Finds WFDB records by scanning for .hea files
2) Reads header metadata for each record (sampling rate, length, channels, etc.)
3) Optionally reads a small preview window of samples (first N seconds) to compute basic stats
4) Writes a manifest file (CSV always; Parquet if available)

IMPORTANT: This script DOES NOT use wfdb's rd_dir parameter.
Instead it passes the FULL record base path (no extension) into wfdb functions,
which works across wfdb versions.

Example:
  data_root = /path/to/mit-bih-arrhythmia-database-1.0.0
  record_base = /path/to/mit-bih-arrhythmia-database-1.0.0/100
  wfdb.rdheader(str(record_base))
  wfdb.rdsamp(str(record_base), sampfrom=0, sampto=...)
"""  # End of the module docstring.

from __future__ import annotations  # Enables modern type hints (like list[Thing]) on older Python versions

import argparse  # Standard library: parses command-line arguments like --data-root
import logging  # Standard library: logs info/warnings/errors with timestamps
from dataclasses import dataclass, asdict  # Standard library: dataclass defines rows; asdict converts to dict
from pathlib import Path  # Standard library: safe path handling across macOS/Linux/Windows
from typing import Iterable, Optional  # Standard library: type hints for readability

import pandas as pd  # Third-party: used to create a DataFrame and write CSV/Parquet

try:  # Start a protected import block
    import wfdb  # Third-party: reads PhysioNet/WFDB headers and sample data
except ImportError as e:  # If wfdb isn't installed, catch the ImportError
    raise SystemExit(  # Exit immediately with a friendly message instead of a confusing crash
        "Missing dependency: wfdb.\n"
        "Install it in your venv with:\n"
        "  pip install wfdb\n"
        f"Original error: {e}"
    )  # End of SystemExit message


@dataclass(frozen=True)  # Define an immutable record (one row in our manifest)
class RecordMeta:  # The schema for one record in the manifest
    record_id: str  # A short identifier like "100" or "subdir/100" (relative-ish label)
    record_base_path: str  # Full path to the record WITHOUT extension (e.g., /.../100)

    n_sig: int  # Number of channels in the record
    fs: float  # Sampling frequency in Hz
    sig_len: int  # Number of samples per channel
    duration_sec: float  # Total duration in seconds (= sig_len / fs)

    sig_names: str  # Comma-separated list of signal/channel names
    units: str  # Comma-separated list of units for each channel

    base_date: Optional[str]  # Optional start date from header (often None)
    base_time: Optional[str]  # Optional start time from header (often None)

    preview_seconds: Optional[float] = None  # How many seconds were previewed (if preview enabled)
    preview_samples: Optional[int] = None  # How many samples were previewed (if preview enabled)
    preview_min: Optional[float] = None  # Minimum sample value over preview (all channels combined)
    preview_max: Optional[float] = None  # Maximum sample value over preview (all channels combined)
    preview_mean: Optional[float] = None  # Mean sample value over preview (all channels combined)


def find_record_bases(data_root: Path) -> Iterable[Path]:  # Function returns Paths to record “bases” (no extension)
    """Scan for .hea files and yield the record base path (no extension)."""  # Human explanation
    for hea_path in data_root.rglob("*.hea"):  # Recursively find all .hea files under data_root
        yield hea_path.with_suffix("")  # Convert /path/100.hea -> /path/100 (record base path)


def read_header(record_base: Path, data_root: Path) -> RecordMeta:  # Parse a record header into RecordMeta
    """Read metadata from the WFDB header for a single record base path."""  # Human explanation

    # record_id is a label relative to the dataset root so your manifest is readable and portable.
    # Example: record_base=/data_root/sub/100 -> record_id="sub/100"
    record_id = str(record_base.relative_to(data_root)).replace("\\", "/")  # Normalize Windows slashes to "/"

    # wfdb.rdheader reads the .hea file and returns a header object.
    # We pass the FULL base path (string) and wfdb figures out .hea next to it.
    header = wfdb.rdheader(str(record_base))  # Read header metadata from record_base.hea

    fs = float(header.fs)  # Sampling rate (Hz) as float
    sig_len = int(header.sig_len)  # Number of samples per channel as int
    duration_sec = (sig_len / fs) if fs > 0 else 0.0  # Compute duration, guard against fs=0

    sig_names = ",".join(header.sig_name) if getattr(header, "sig_name", None) else ""  # Channel names -> CSV string
    units = ",".join(getattr(header, "units", []) or [])  # Units list -> CSV string (safe even if missing)

    base_date = str(getattr(header, "base_date", None)) if getattr(header, "base_date", None) else None  # Date or None
    base_time = str(getattr(header, "base_time", None)) if getattr(header, "base_time", None) else None  # Time or None

    return RecordMeta(  # Build and return a RecordMeta row
        record_id=record_id,  # Store the record label
        record_base_path=str(record_base),  # Store the full base path (no extension)
        n_sig=int(header.n_sig),  # Number of channels
        fs=fs,  # Sampling rate
        sig_len=sig_len,  # Samples per channel
        duration_sec=float(duration_sec),  # Total duration in seconds
        sig_names=sig_names,  # Comma-separated channel names
        units=units,  # Comma-separated units
        base_date=base_date,  # Optional date
        base_time=base_time,  # Optional time
    )  # End return RecordMeta


def add_preview_stats(meta: RecordMeta, max_seconds: float) -> RecordMeta:  # Add preview statistics by reading samples
    """Read first N seconds of samples and compute min/max/mean over that preview."""  # Human explanation

    import numpy as np  # Local import so numpy is only required if preview is used (nice for minimal installs)

    fs = meta.fs  # Copy sampling rate into a local variable for clarity
    sig_len = meta.sig_len  # Copy signal length into a local variable for clarity

    preview_samples = int(fs * max_seconds)  # Convert seconds -> samples
    preview_samples = min(preview_samples, sig_len)  # Cap preview so we never read beyond the record

    # Read samples: signals is a 2D array of shape (preview_samples, n_sig).
    # We pass the full record base path (no extension) stored in meta.record_base_path.
    signals, _fields = wfdb.rdsamp(  # Call wfdb to read sample data
        meta.record_base_path,  # Full base path string (e.g., /.../100)
        sampfrom=0,  # Start at sample 0
        sampto=preview_samples,  # Stop at preview_samples
    )  # End wfdb.rdsamp call

    flat = signals.astype("float64").ravel()  # Flatten all channels into one vector for simple aggregate stats
    if flat.size == 0:  # If no samples were returned, just return meta unchanged
        return meta  # Nothing to compute

    updated = asdict(meta)  # Convert immutable dataclass -> dict so we can update fields
    updated["preview_seconds"] = (preview_samples / fs) if fs > 0 else None  # Store preview duration
    updated["preview_samples"] = int(preview_samples)  # Store preview samples
    updated["preview_min"] = float(np.min(flat))  # Min sample value
    updated["preview_max"] = float(np.max(flat))  # Max sample value
    updated["preview_mean"] = float(np.mean(flat))  # Mean sample value

    return RecordMeta(**updated)  # Rebuild and return a new immutable RecordMeta with preview stats


def write_manifest(records: list[RecordMeta], out_dir: Path) -> None:  # Write outputs to disk
    """Write manifest.csv always; write manifest.parquet if parquet engine exists."""  # Human explanation

    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

    df = pd.DataFrame([asdict(r) for r in records])  # Convert records into a DataFrame

    csv_path = out_dir / "manifest.csv"  # Path for CSV output
    df.to_csv(csv_path, index=False)  # Write CSV manifest

    # Try writing Parquet, but don’t fail the whole run if pyarrow/fastparquet isn’t installed.
    parquet_path = out_dir / "manifest.parquet"  # Path for Parquet output
    try:  # Attempt parquet write
        df.to_parquet(parquet_path, index=False)  # Write Parquet manifest (requires pyarrow or fastparquet)
    except Exception as e:  # If Parquet fails, log and continue
        logging.getLogger("ingest").warning(  # Use logger to warn instead of crashing
            "Could not write Parquet (%s). CSV is still written at %s.", e, csv_path
        )  # End warning


def build_parser() -> argparse.ArgumentParser:  # Construct CLI argument parser
    """Define the command-line interface for this script."""  # Human explanation
    p = argparse.ArgumentParser(description="Ingest a PhysioNet WFDB dataset into a manifest.")  # Create parser
    p.add_argument("--data-root", required=True, help="Directory containing WFDB .hea/.dat files.")  # Dataset folder
    p.add_argument("--out", required=True, help="Output directory for manifest files.")  # Output folder
    p.add_argument("--read-samples", action="store_true", help="If set, compute preview stats from samples.")  # Flag
    p.add_argument("--max-seconds", type=float, default=10.0, help="Seconds to preview if --read-samples is set.")  # Preview length
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N records (for testing).")  # Limit records
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")  # Logging level
    return p  # Return the configured parser


def main() -> int:  # Program entrypoint that returns a shell-style exit code
    args = build_parser().parse_args()  # Parse command-line arguments into args object

    logging.basicConfig(  # Configure logging once for the whole program
        level=getattr(logging, args.log_level),  # Convert "INFO" -> logging.INFO, etc.
        format="%(asctime)s %(levelname)s %(message)s",  # Timestamped logs
    )  # End logging config

    log = logging.getLogger("ingest")  # Create/retrieve a named logger for this script

    data_root = Path(args.data_root).expanduser().resolve()  # Convert data-root to absolute normalized path
    out_dir = Path(args.out).expanduser().resolve()  # Convert out to absolute normalized path

    log.info("CWD: %s", Path.cwd())  # Print current working directory (helps debug relative-path issues)
    log.info("Data root: %s", data_root)  # Log where we're reading from
    log.info("Output dir: %s", out_dir)  # Log where we're writing to

    if not data_root.exists():  # If the dataset folder doesn't exist, stop early with a clear error
        log.error("data-root does not exist: %s", data_root)  # Log error
        return 2  # Non-zero exit code signals failure to the shell

    record_bases = sorted(find_record_bases(data_root))  # Find and sort all record base paths
    if args.limit > 0:  # If user requested a limit
        record_bases = record_bases[: args.limit]  # Keep only first N record bases

    if not record_bases:  # If we found no .hea files at all
        log.error("No .hea files found under: %s", data_root)  # Log why we can't proceed
        return 3  # Exit code for "no records"

    log.info("Found %d records (.hea files).", len(record_bases))  # Log count

    records: list[RecordMeta] = []  # Prepare list to store RecordMeta results
    failures = 0  # Count how many records failed to parse/read

    for record_base in record_bases:  # Loop over each record base path (e.g., /.../100)
        try:  # Wrap per-record work so one failure doesn't stop the entire dataset
            meta = read_header(record_base, data_root)  # Read header metadata into RecordMeta
            if args.read_samples:  # If preview stats enabled
                meta = add_preview_stats(meta, max_seconds=args.max_seconds)  # Read first N seconds and compute stats
            records.append(meta)  # Add meta row to list
            log.debug("OK: %s", meta.record_id)  # Debug log for successful record
        except Exception as e:  # Catch any exception for that record
            failures += 1  # Increment failure count
            log.warning("Failed record %s: %s", record_base, e)  # Log which record failed and why

    if not records:  # If everything failed
        log.error("All records failed to ingest. Check dataset integrity and wfdb installation.")  # Log error
        return 4  # Exit code for "total failure"

    write_manifest(records, out_dir)  # Write output manifests to disk

    log.info("Done. OK=%d Failed=%d", len(records), failures)  # Final summary line
    return 0  # Success exit code


if __name__ == "__main__":  # Only runs when you execute the script directly (not when importing it)
    raise SystemExit(main())  # Call main() and exit with its return code
