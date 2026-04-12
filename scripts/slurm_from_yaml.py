#!/usr/bin/env python3
"""Submit a SLURM job from a YAML config file.

Usage:
    python scripts/slurm_from_yaml.py <config.yaml> <script.py> [--dry-run]

The YAML must have three top-level keys:
    slurm:  SLURM #SBATCH directives (snake_case keys mapped to --kebab-case flags)
    env:    environment variables exported before srun
    args:   CLI arguments forwarded to <script.py> (booleans become bare flags)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import yaml


SLURM_KEY_MAP = {
    "job_name":       "job-name",
    "output":         "output",
    "error":          "error",
    "nodes":          "nodes",
    "ntasks":         "ntasks",
    "cpus_per_task":  "cpus-per-task",
    "mem":            "mem",
    "time":           "time",
    "partition":      "partition",
    "gres":           "gres",
    "nodelist":       "nodelist",
    "exclude":        "exclude",
    "account":        "account",
    "qos":            "qos",
    "mail_user":      "mail-user",
    "mail_type":      "mail-type",
}


def build_sbatch_script(cfg: dict, script: str) -> str:
    lines = ["#!/bin/bash"]

    # ── SLURM directives ─────────────────────────────────────────────────────
    for key, value in cfg.get("slurm", {}).items():
        if key.startswith("#"):
            continue
        sbatch_flag = SLURM_KEY_MAP.get(key, key.replace("_", "-"))
        lines.append(f"#SBATCH --{sbatch_flag}={value}")

    lines += [
        "",
        "set -e",
        # Batch jobs often lack ~/.local/bin; uv is typically installed there.
        'export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"',
        "",
        'echo "Running on $(hostname)"',
        'echo "Running on partition: $SLURM_JOB_PARTITION"',
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "Node list: $SLURM_NODELIST"',
        'echo "GPUs: $SLURM_GPUS"',
        "",
        "nvidia-smi",
        "",
    ]

    # ── Environment variables ────────────────────────────────────────────────
    for key, value in cfg.get("env", {}).items():
        lines.append(f"export {key}={value}")

    if cfg.get("env"):
        lines.append("")

    # ── srun command (uv if present, else project .venv — same as trainISO.sh) ─
    arg_lines: list[str] = []
    for key, value in cfg.get("args", {}).items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arg_lines.append(f"    --{flag}")
        elif value is not None:
            arg_lines.append(f"    --{flag} {value}")

    arg_suffix = ""
    if arg_lines:
        arg_suffix = " \\\n" + " \\\n".join(arg_lines)

    lines += [
        'cd "${SLURM_SUBMIT_DIR:-.}"',
        "",
        "if command -v uv >/dev/null 2>&1; then",
        f"  srun uv run {script}{arg_suffix}",
        "elif [[ -x .venv/bin/python ]]; then",
        f"  srun .venv/bin/python {script}{arg_suffix}",
        "else",
        '  echo "ERROR: neither uv nor .venv/bin/python found. Run: uv sync (in repo)" >&2',
        "  exit 1",
        "fi",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("script", help="Python script to run (e.g. trainISO.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated sbatch script without submitting")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    script_content = build_sbatch_script(cfg, args.script)

    if args.dry_run:
        print(script_content)
        return

    # Write to a named temp file so sbatch can read it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
        tmp.write(script_content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(["sbatch", tmp_path], check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"sbatch failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
