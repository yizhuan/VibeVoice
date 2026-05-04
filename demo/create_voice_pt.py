import argparse
import os

import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a voice audio file (.wav/.mp3/.flac/...) into a VibeVoice-compatible .pt tensor file."
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to source audio file (e.g. demo/voices/my_voice.wav)",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default=None,
        help="Output .pt path. Defaults to input basename with .pt extension.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=24000,
        help="Target sampling rate for VibeVoice (default: 24000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_audio):
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")

    output_pt = args.output_pt
    if output_pt is None:
        input_stem, _ = os.path.splitext(args.input_audio)
        output_pt = f"{input_stem}.pt"

    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)

    wav, _ = librosa.load(args.input_audio, sr=args.sampling_rate, mono=True)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)

    torch.save(wav_tensor, output_pt)

    duration = wav_tensor.numel() / float(args.sampling_rate)
    print(f"Saved: {output_pt}")
    print(f"Shape: {tuple(wav_tensor.shape)}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Duration: {duration:.2f} s")


if __name__ == "__main__":
    main()
