# AGENTS.md

These instructions apply to the entire repository.

## Python and dependencies

- Use `uv` for every Python command. Run scripts with `uv run python <script>` and modules with `uv run python -m <module>`.
- Do not invoke `python` or `python3` directly.
- Install the project with `uv pip install -e .`.
- Add or install Python packages through `uv`; do not use bare `pip`.
- The project requires Python 3.10 or newer.

## Repository layout

- `vibevoice/` contains the package implementation.
- `demo/` contains CLI and Gradio entry points, example scripts, and voice assets.
- `docs/` contains focused user documentation.
- `outputs/` is generated output and should not be treated as source.

## Development guidelines

- Keep changes focused and consistent with nearby code.
- Preserve public APIs and model compatibility unless the task explicitly requires a change.
- Do not commit generated audio, downloaded model weights, caches, or local environment files.
- Speaker scripts use `Speaker N: text` labels. Multi-speaker inference supports up to four speakers.
- Prefer small helpers for input transformation so behavior can be validated without loading model weights.

## Validation

- There is currently no repository-wide automated test suite. Run the narrowest relevant check for the changed code.
- Compile a changed Python file with `uv run python -m py_compile <path>`.
- Run a demo with `uv run python demo/<script>.py ...`.
- Avoid model inference as a routine validation step because it may require large downloads and GPU resources; use it when the task specifically concerns runtime inference behavior and the required assets are available.