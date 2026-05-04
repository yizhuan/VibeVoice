import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
VOICES_DIR = DEMO_DIR / "voices"
TEXT_EXAMPLES_DIR = DEMO_DIR / "text_examples"
INFERENCE_SCRIPT = DEMO_DIR / "inference_from_file.py"

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def discover_voice_files() -> List[Tuple[str, str]]:
    if not VOICES_DIR.exists():
        return []
    results: List[Tuple[str, str]] = []
    for p in sorted(VOICES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
            results.append((p.stem, str(p)))
    return results


def discover_text_examples() -> List[str]:
    if not TEXT_EXAMPLES_DIR.exists():
        return []
    return [
        str(p.relative_to(REPO_ROOT))
        for p in sorted(TEXT_EXAMPLES_DIR.glob("*.txt"))
        if p.is_file()
    ]


VOICE_ITEMS = discover_voice_files()
VOICE_MAP = {name: path for name, path in VOICE_ITEMS}
VOICE_NAMES = list(VOICE_MAP.keys())
TEXT_EXAMPLES = discover_text_examples()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def preview_speaker(speaker_name: str):
    if not speaker_name:
        return None, "Pick a speaker to preview."
    audio_path = VOICE_MAP.get(speaker_name)
    if not audio_path:
        return None, f"Speaker '{speaker_name}' not found in demo/voices."
    return audio_path, f"Previewing {speaker_name}."


def load_script_preview(txt_path: str):
    if not txt_path:
        return ""
    resolved = _resolve_path(txt_path)
    if not resolved.exists():
        return f"File not found: {resolved}"
    try:
        return resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Failed to read file: {exc}"


def resolve_script_path(sample_txt_path: str, custom_txt_path: str, uploaded_txt_path: str) -> Tuple[Path, str]:
    # Priority: uploaded file > custom path > sample dropdown
    if uploaded_txt_path:
        resolved = Path(uploaded_txt_path).expanduser().resolve()
        if resolved.exists():
            return resolved, "uploaded"
        return None, f"uploaded file not found: {resolved}"

    if custom_txt_path and custom_txt_path.strip():
        resolved = _resolve_path(custom_txt_path.strip())
        if resolved.exists():
            return resolved, "custom"
        return None, f"custom txt file not found: {resolved}"

    if sample_txt_path and sample_txt_path.strip():
        resolved = _resolve_path(sample_txt_path.strip())
        if resolved.exists():
            return resolved, "sample"
        return None, f"sample txt file not found: {resolved}"

    return None, "no script source provided"


def load_script_preview_from_inputs(sample_txt_path: str, custom_txt_path: str, uploaded_txt_path: str):
    resolved, source = resolve_script_path(sample_txt_path, custom_txt_path, uploaded_txt_path)
    if resolved is None:
        return f"Error: {source}"

    try:
        body = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Failed to read file ({resolved}): {exc}"

    return f"[source={source}] {resolved}\n\n{body}"


def run_inference(
    model_path: str,
    sample_txt_path: str,
    custom_txt_path: str,
    uploaded_txt_path: str,
    selected_speakers: List[str],
    output_dir: str,
    device: str,
    checkpoint_path: str,
    disable_prefill: bool,
    cfg_scale: float,
    seed: float,
):
    txt_resolved, source = resolve_script_path(sample_txt_path, custom_txt_path, uploaded_txt_path)
    if txt_resolved is None:
        return None, f"Error: {source}"

    if not selected_speakers:
        return None, "Error: select at least one speaker."
    if len(selected_speakers) > 4:
        return None, "Error: at most 4 speakers are supported."

    output_resolved = _resolve_path(output_dir or "outputs")
    output_resolved.mkdir(parents=True, exist_ok=True)
    speaker_suffix = "_".join(selected_speakers)

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--model_path",
        model_path,
        "--txt_path",
        str(txt_resolved),
        "--output_dir",
        str(output_resolved),
        "--speaker_names",
        *selected_speakers,
        "--output_suffix",
        speaker_suffix,
        "--cfg_scale",
        str(cfg_scale),
    ]

    if device:
        cmd += ["--device", device]
    if checkpoint_path:
        cmd += ["--checkpoint_path", checkpoint_path]
    if disable_prefill:
        cmd += ["--disable_prefill"]
    if seed is not None and seed >= 0:
        cmd += ["--seed", str(int(seed))]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return None, f"Failed to launch inference: {exc}"

    logs = []
    logs.append(f"Command: {' '.join(cmd)}")
    logs.append(f"Script source: {source}")
    logs.append(f"Script path: {txt_resolved}")
    logs.append(f"Exit code: {proc.returncode}")
    if proc.stdout:
        logs.append("\n--- stdout ---\n" + proc.stdout.strip())
    if proc.stderr:
        logs.append("\n--- stderr ---\n" + proc.stderr.strip())

    if proc.returncode != 0:
        return None, "\n".join(logs)

    generated_path = None
    if proc.stdout:
        for line in proc.stdout.splitlines()[::-1]:
            marker = "Saved output to "
            if marker in line:
                generated_path = line.split(marker, 1)[1].strip()
                break

    if generated_path and Path(generated_path).exists():
        logs.append(f"\nGenerated file: {generated_path}")
        return generated_path, "\n".join(logs)

    # Fallback: pick latest generated wav for this script stem.
    txt_stem = txt_resolved.stem
    candidates = sorted(output_resolved.glob(f"{txt_stem}_generated*.wav"), key=lambda p: p.stat().st_mtime)
    if candidates:
        generated_path = str(candidates[-1])
        logs.append(f"\nGenerated file (fallback): {generated_path}")
        return generated_path, "\n".join(logs)

    logs.append("\nWarning: no generated output file was detected.")
    return None, "\n".join(logs)


def build_ui() -> gr.Blocks:
    default_txt = TEXT_EXAMPLES[0] if TEXT_EXAMPLES else "demo/text_examples/1p_abs.txt"
    default_preview_speaker = VOICE_NAMES[0] if VOICE_NAMES else None
    default_preview_audio = VOICE_MAP.get(default_preview_speaker) if default_preview_speaker else None
    default_preview_status = (
        f"Previewing {default_preview_speaker}." if default_preview_speaker else "Pick a speaker to preview."
    )

    with gr.Blocks(title="VibeVoice File Inference UI") as demo:
        gr.Markdown("# VibeVoice Inference UI")
        gr.Markdown(
            "Select one to four speakers, preview voice samples, run file-based inference, and preview the generated output."
        )

        with gr.Row():
            with gr.Column(scale=3):
                model_path = gr.Textbox(label="Model Path", value="vibevoice/VibeVoice-1.5B")

                with gr.Group():
                    gr.Markdown("### Script Preview")
                    sample_txt_path = gr.Dropdown(
                        choices=TEXT_EXAMPLES,
                        value=default_txt,
                        label="Text Script Path (Sample)",
                        allow_custom_value=True,
                    )
                    custom_txt_path = gr.Textbox(
                        label="Text Script Path (Custom)",
                        value="",
                        placeholder="Enter a relative or absolute .txt path",
                    )
                    uploaded_txt_path = gr.File(
                        label="Or Browse Text File",
                        file_count="single",
                        file_types=[".txt", ".md"],
                        type="filepath",
                    )
                    load_script_btn = gr.Button("Load Script Preview")
                    script_preview = gr.Textbox(label="Script Preview", lines=10)

                with gr.Group():
                    gr.Markdown("### Speakers")
                    selected_speakers = gr.Dropdown(
                        choices=VOICE_NAMES,
                        value=VOICE_NAMES[:1],
                        multiselect=True,
                        max_choices=4,
                        label="Speakers (up to 4)",
                    )

                with gr.Group():
                    gr.Markdown("### Speaker to Preview")
                    preview_speaker_name = gr.Dropdown(
                        choices=VOICE_NAMES,
                        value=default_preview_speaker,
                        label="Speaker to Preview",
                    )
                    preview_audio = gr.Audio(label="Speaker Preview", type="filepath", value=default_preview_audio)
                    preview_status = gr.Textbox(label="Preview Status", lines=2, value=default_preview_status)

            with gr.Column(scale=2):
                output_dir = gr.Textbox(label="Output Directory", value="outputs")
                device = gr.Dropdown(
                    choices=["", "cuda", "mps", "cpu"],
                    value="",
                    label="Device (optional)",
                )
                checkpoint_path = gr.Textbox(label="Checkpoint Path (optional)", value="")
                disable_prefill = gr.Checkbox(label="Disable Prefill / Voice Cloning", value=False)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=3.0, step=0.1, value=1.3)
                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                run_btn = gr.Button("Generate", variant="primary")

                generated_audio = gr.Audio(label="Generated Audio", type="filepath")
                run_logs = gr.Textbox(label="Run Logs", lines=18)

        load_script_btn.click(
            load_script_preview_from_inputs,
            inputs=[sample_txt_path, custom_txt_path, uploaded_txt_path],
            outputs=[script_preview],
        )
        preview_speaker_name.change(
            preview_speaker,
            inputs=[preview_speaker_name],
            outputs=[preview_audio, preview_status],
        )
        run_btn.click(
            run_inference,
            inputs=[
                model_path,
                sample_txt_path,
                custom_txt_path,
                uploaded_txt_path,
                selected_speakers,
                output_dir,
                device,
                checkpoint_path,
                disable_prefill,
                cfg_scale,
                seed,
            ],
            outputs=[generated_audio, run_logs],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio UI for demo/inference_from_file.py")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
