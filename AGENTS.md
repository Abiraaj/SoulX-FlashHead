# Repository Guidelines

## Project Structure & Module Organization

- `generate_video.py` is the command-line inference entry point; `gradio_app.py` and `gradio_app_streaming.py` provide browser demos.
- `flash_head/inference.py` connects preprocessing and generation. Core pipelines, model modules, and distributed helpers live under `flash_head/src/`.
- `flash_head/audio_analysis/` handles audio features; `flash_head/utils/` contains image and face utilities. `wan/` and `ltx_video/` contain supporting model components.
- `flash_head/configs/infer_params.yaml` defines inference defaults. `examples/` contains sample inputs; `assets/` holds documentation media. Download checkpoints into `models/`.

## Build, Test, and Development Commands

Run commands from the repository root; configuration paths are relative to it. Follow `README.md` for the Python 3.10 environment, CUDA PyTorch installation, FlashAttention, FFmpeg, and checkpoint downloads.

- `pip install -r requirements.txt` installs application dependencies after PyTorch setup.
- `bash inference_script_single_gpu_lite.sh` runs Lite inference with sample inputs.
- `bash inference_script_single_gpu_pro.sh` runs single-GPU Pro inference.
- `bash inference_script_multi_gpu_pro.sh` runs distributed Pro inference.
- `python gradio_app.py` starts the standard demo; `python gradio_app_streaming.py` starts the single-GPU streaming demo.

The shell launchers require Bash and CUDA-capable hardware; inspect GPU selections and model paths before running. There is no separate package build command.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Match surrounding Python style and preserve upstream copyright notices. Keep inference defaults in YAML and reusable logic inside `flash_head/`. No formatter or linter configuration is supplied; avoid unrelated formatting changes.

## Testing Guidelines

No automated test suite, test framework, or coverage threshold is provided. Validate affected inference paths with the sample image and audio, checking playable output, audio synchronization, and streaming behavior when relevant. Record model variant, GPU, dependency versions, seed, command, and results. If adding automated tests, use `tests/test_<feature>.py` and document their runner and hardware requirements.

## Commit & Pull Request Guidelines

This downloaded snapshot has no Git history to establish commit conventions. Use concise imperative subjects, such as `Fix streaming audio chunk alignment`. Keep changes focused. PRs should explain behavior changes, link relevant issues, and report validation commands and results; include screenshots or short clips for demo changes. Keep checkpoints, generated videos, credentials, and local environments out of commits; respect existing `.gitignore` exclusions.
