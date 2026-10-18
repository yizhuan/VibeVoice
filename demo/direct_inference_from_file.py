#!/usr/bin/env python3
"""Run VibeVoice inference and save to a caller-specified output path.

This is the automation-friendly contract we use in the research pipeline:
    - the caller tells us the exact output WAV path
    - the script generates the audio and writes to that exact file
    - no directory scanning or filename guessing is required afterward
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_output_path(output_path: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_script(script: str, *, model_path: str, txt_path: str, output_dir: str,
               speaker_name: str | None = None, speaker_names: list[str] | None = None,
               device: str, cfg_scale: float, seed: int | None, ddpm_steps: int | None,
               stream: bool = False) -> tuple[Path | None, str]:
    cmd = [sys.executable, str(REPO_ROOT / script), "--model_path", model_path,
           "--txt_path", txt_path, "--output_dir", output_dir,
           "--device", device, "--cfg_scale", str(cfg_scale)]

    if stream:
        if speaker_name is None:
            raise ValueError("--speaker_name is required when --streaming is used")
        cmd += ["--speaker_name", speaker_name]
        if ddpm_steps is not None:
            cmd += ["--ddpm_steps", str(ddpm_steps)]
    else:
        if not speaker_names:
            raise ValueError("--speaker_names is required unless --streaming is used")
        cmd += ["--speaker_names", *speaker_names]

    if seed is not None:
        cmd += ["--seed", str(seed)]

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    logs = [
        f"Command: {' '.join(cmd)}",
        f"Script: {script}",
        f"Exit code: {proc.returncode}",
    ]
    if stdout:
        logs.append("--- stdout ---\n" + stdout)
    if stderr:
        logs.append("--- stderr ---\n" + stderr)

    generated = None
    if stdout:
        for line in stdout.splitlines()[::-1]:
            marker = "Saved output to "
            if marker in line:
                generated = Path(line.split(marker, 1)[1].strip())
                break

    if generated and generated.exists():
        return generated, "\n".join(logs)

    txt_stem = Path(txt_path).stem
    candidates = []
    for p in Path(output_dir).glob(f"{txt_stem}_generated*.wav"):
        if p.is_file():
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0], "\n".join(logs)

    if proc.returncode != 0:
        raise RuntimeError("\n".join(logs))

    return None, "\n".join(logs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt_path", required=True, help="Text script to synthesize")
    parser.add_argument("--output_path", required=True, help="Exact WAV path to write")
    parser.add_argument("--model_path", default="vibevoice/VibeVoice-1.5B", help="Model path or repo ID")
    parser.add_argument("--speaker_name", help="Single speaker name for streaming model")
    parser.add_argument("--speaker_names", nargs="+", help="Speaker names for standard model")
    parser.add_argument("--device", default=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"), help="cuda | cpu | mps")
    parser.add_argument("--cfg_scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ddpm_steps", type=int, default=5)
    parser.add_argument("--streaming", action="store_true", help="Use the streaming 0.5B model path")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--disable_prefill", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_output_path(args.output_path)
    if output_path.exists():
        output_path.unlink()

    txt_path = str(Path(args.txt_path).expanduser())
    if not Path(txt_path).is_file():
        raise SystemExit(f"txt file not found: {txt_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="vv-direct-") )
    try:
        if args.streaming:
            script = "demo/streaming_inference_from_file.py"
            generated, logs = run_script(
                script,
                model_path=args.model_path,
                txt_path=txt_path,
                output_dir=str(temp_dir),
                speaker_name=args.speaker_name,
                speaker_names=None,
                device=args.device,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                ddpm_steps=args.ddpm_steps,
                stream=True,
            )
        else:
            script = "demo/inference_from_file.py"
            generated, logs = run_script(
                script,
                model_path=args.model_path,
                txt_path=txt_path,
                output_dir=str(temp_dir),
                speaker_name=None,
                speaker_names=(args.speaker_names or []),
                device=args.device,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                ddpm_steps=args.ddpm_steps,
                stream=False,
            )

        if generated is None or not generated.exists():
            raise SystemExit(f"No generated output file was produced.\n{logs}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if generated.resolve() != output_path.resolve():
            shutil.copy2(generated, output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise SystemExit(f"Final output was not written: {output_path}\n{logs}")

        print(f"Saved output to {output_path}")
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
