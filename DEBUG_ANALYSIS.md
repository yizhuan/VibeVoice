# VibeVoice "preview audio contamination" — root cause and fix

## What the contaminated audio actually contains

Transcribing `~/.research-video/github/daily/2026-09-08/audio/85-notable-shifts.wav`
(whisper.cpp) shows paragraph 1 spoken correctly, then, on loop:

> "...the public transport firm have confirmed no additional services will run on
> St. Patrick's Day later this month. Thousands are expected in the city on Tuesday,
> 17th March..."

That text is the spoken content of the voice reference `demo/voices/en-Yizhuan_man.wav`.
It is not a cache, tmp or preview file. VibeVoice re-spoke its in-context voice prompt.

## Why it happens at the end of the first paragraph

`prepare_script()` (solo profile) emitted one `Speaker 1:` line per paragraph. In the
model prompt the voice sample is also presented as `Speaker 0: <speech>`. At every
paragraph boundary the model must pick "the next Speaker 0 turn", and it sometimes
locks onto the voice-prompt speech (regurgitation) or skips a paragraph.
A/B on the failing section (upstream generation code, 3 seeds each):

| layout | complete | dropped a paragraph | voice-prompt leak |
|---|---|---|---|
| one `Speaker 1:` per paragraph | 1/3 | 2/3 | seen in production run |
| single `Speaker 1:` turn | 3/3 | 0/3 | 0/3 |

## Why the in-memory state theory was wrong

The pipeline runs each section in a fresh `direct_inference_from_file.py` process,
so no generation state can survive between sections. The experimental scheduler /
negative-cache / streamer resets added to `vibevoice/modular/*` were reverted to
upstream; the per-step negative-cache rebuild had degraded CFG (this was the
"quality dropped" observation). Only the `--seed` → `torch.Generator` plumbing for
diffusion noise is kept.

## Fix (nebula-research-radar `skills/narration-to-audio/scripts/synthesize.py`)

1. Solo narration is now a single `Speaker 1:` turn (paragraphs joined with spaces).
2. A whisper.cpp (`ggml-base.en`) fidelity gate: script-word recall and transcript
   precision. Good clips score ≥ 0.95; a dropped paragraph ≈ 0.7 recall; voice-prompt
   regurgitation ≈ 0.65 precision. Below 0.85 on either triggers the existing
   retry-with-new-seed, then `needs_review`. Values are recorded in the audio sidecar
   as `asr_recall` / `asr_precision`. The duration check alone had passed the bad
   clip (84.8 s vs 73.7 s expected, inside the 35 % tolerance).

Regenerated `85-notable-shifts` through the pipeline: 68.0 s, recall 0.96,
precision 0.95, status ok.
