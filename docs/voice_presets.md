# Repo

git clone https://github.com/vibevoice-community/VibeVoice.git

---

# Voice Presets

This document explains how voice presets work for each model, how they are stored, and how inference uses them.

---

## Two Different Models, Two Different Formats

| Model | Voice file format | What it contains |
|---|---|---|
| VibeVoice-1.5B | `.wav` or `.pt` waveform tensor | Raw audio samples at 24 kHz |
| VibeVoice-Realtime-0.5B | `.pt` KV-cache dict | Pre-computed transformer hidden states |

The two formats are completely different. The `.pt` extension means different things depending on which model uses it.

---

## VibeVoice-1.5B: Audio Waveform Presets

### What is stored

A `.pt` file for the 1.5B model is simply a serialised PyTorch tensor:

```python
torch.FloatTensor of shape (num_samples,)  # mono, 24 kHz
```

It is interchangeable with a `.wav` file. The only reason to prefer `.pt` is to avoid the WAV header overhead and guarantee the correct sample rate on load.

### Creating a preset

`demo/create_voice_pt.py` converts any audio file to the expected format:

```bash
python demo/create_voice_pt.py \
    --input_audio demo/voices/my_voice.wav \
    --output_pt  demo/voices/my_voice.pt \
    --sampling_rate 24000
```

Internally it runs `librosa.load(..., sr=24000, mono=True)` and calls `torch.save()`.

### How inference uses it

`demo/inference_from_file.py` contains a `VoiceMapper` class that scans `demo/voices/` for both `.wav` and `.pt` files. When a base name exists in both formats, `.pt` is preferred:

```python
# Preference map: higher value wins
supported_exts = {'.wav': 1, '.pt': 2}
```

The resolved path is passed to `VibeVoiceProcessor`:

```python
inputs = processor(
    text=[full_script],
    voice_samples=[voice_samples],   # list of .wav or .pt paths
    ...
)
```

The processor loads the audio, normalises its dB level, resamples if needed, and encodes it through the acoustic tokeniser. These acoustic tokens are prepended to the input sequence as the voice-conditioning "prefill". Setting `--disable_prefill` skips this step, generating speech without any voice reference.

### Voice name resolution

`VoiceMapper` tries exact match first, then case-insensitive substring match. File names follow the convention `<lang>-<Name>_<gender>.wav`, e.g. `en-Alice_woman.wav`. The aliases `Alice`, `alice`, and `woman` all resolve to the same file.

---

## VibeVoice-Realtime-0.5B: KV-Cache Presets

### Architecture background

The Realtime model has two transformer stacks:

- **`language_model`** (`lm`) — the upper text LM layers
- **`tts_language_model`** (`tts_lm`) — the lower TTS-specific layers

Classifier-free guidance (CFG) requires running both a conditioned and an unconditioned path, giving four sets of cached states per preset:

| Key | Description |
|---|---|
| `lm` | Conditioned LM hidden states / KV cache |
| `tts_lm` | Conditioned TTS-LM hidden states / KV cache |
| `neg_lm` | Unconditioned LM hidden states / KV cache |
| `neg_tts_lm` | Unconditioned TTS-LM hidden states / KV cache |

Each value is a dict with a `last_hidden_state` tensor of shape `(1, T, hidden_size)` plus the attention key/value tensors.

### What is stored

A streaming preset is a dict saved with `torch.save`:

```python
preset = {
    "lm":         {...},   # BaseModelOutputWithPast
    "tts_lm":     {...},   # BaseModelOutputWithPast
    "neg_lm":     {...},
    "neg_tts_lm": {...},
}
```

These are found in `demo/voices/streaming_model/`, e.g. `en-Emma_woman.pt`.

### Why creating new presets from audio is non-trivial

The preset KV caches are produced by running the model's acoustic encoder on a reference audio clip. However, **the acoustic encoder weights are not included in the public `microsoft/VibeVoice-Realtime-0.5B` checkpoint**. This means:

- You cannot run the encoder on new audio to build a fresh `tts_lm` cache.
- Gradient inversion (reconstructing the encoder input from the decoder Jacobian) is infeasible because the Jacobian at `z = 0` is ≈ 0.

The practical workaround is to **reuse the `tts_lm` and `neg_tts_lm` from an existing preset** (which determines the voice timbre) while optionally replacing `lm` / `neg_lm` (which are conditioned on a text transcript and control prosodic style). This was implemented in the prior session as `--copy_voice_from` in a preset-creation helper script.

### How inference uses the preset

`demo/streaming_inference_from_file.py` loads the preset and passes it straight to `generate()`:

```python
all_prefilled_outputs = torch.load(voice_sample, map_location=device, weights_only=False)

inputs = processor.process_input_with_cached_prompt(
    text=full_script,
    cached_prompt=all_prefilled_outputs,
    ...
)

outputs = model.generate(
    **inputs,
    all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
    ...
)
```

Inside `generate()`:

1. The four cached outputs are unpacked from `all_prefilled_outputs`.
2. `_update_model_kwargs_for_generation` restores the KV caches for each of the four transformer paths, so the next forward pass continues from the end of the voice priming sequence rather than from scratch.
3. The processor builds placeholder `input_ids` whose length matches `lm.last_hidden_state.size(1)` and `tts_lm.last_hidden_state.size(1)`, so position IDs and attention masks are correct.
4. Generation then proceeds token-by-token with windowed text feed-in and diffusion-based acoustic decoding.

### Token count accounting

Because the prefilled voice tokens are not in `input_ids` directly, the `generated_tokens` metric subtracts them:

```python
generated_tokens = output_tokens - input_tokens \
    - all_prefilled_outputs['tts_lm']['last_hidden_state'].size(1)
```

---

## Summary

```
demo/voices/
├── en-Alice_woman.wav          # 1.5B: raw audio, used by VoiceMapper
├── my_voice.wav                # 1.5B: custom voice (original WAV)
├── my_voice.pt                 # 1.5B: converted by create_voice_pt.py
│                               #       (.pt wins over .wav for same name)
└── streaming_model/
    ├── en-Emma_woman.pt        # Realtime-0.5B: KV-cache dict
    ├── en-Carter_man.pt        #   keys: lm, tts_lm, neg_lm, neg_tts_lm
    └── ...
```

**To add a custom voice for VibeVoice-1.5B**: place a `.wav` (or run `create_voice_pt.py` to produce a `.pt`) in `demo/voices/` and pass its base name as `--speaker_names`.

**To add a custom voice for Realtime-0.5B**: copy an existing `.pt` preset from `demo/voices/streaming_model/` and, if the acoustic encoder becomes available, replace the `tts_lm`/`neg_tts_lm` keys with states computed from the target voice audio.
