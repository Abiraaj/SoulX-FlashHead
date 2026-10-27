# POC validation

- `python -m py_compile gradio_app.py`: PASS (Python 3.12.3).
- Earlier media and launch-adapter checks: PASS, covering valid uploads, invalid/oversized images and files, missing audio streams, zero/overlong/nonfinite durations, arbitrary path rejection, subprocess argument safety, cleanup boundaries, and launch-adapter enforcement. ffprobe responses were mocked.
- Live Gradio launch, share authentication, FFmpeg processing, and CUDA inference were not run; Gradio and ffprobe are unavailable in this local environment.
- Keep the pinned [Gradio 5.50.0](https://pypi.org/project/gradio/5.50.0/). It has no run-history feature or `run_history` launch argument; a narrow launch adapter accepts only `run_history=False`. Gradio 6.30.0 requires a Hugging Face Hub version incompatible with the pinned Transformers version.
- Credential defaults, environment overrides, and empty-credential rejection: PASS (3 checks). The previous missing/short credential checks are superseded; the 16-character minimum was removed as requested.
- Built-in credentials are now available; optionally override them with `APP_USERNAME` and `APP_PASSWORD`. Run `python gradio_app.py` from the repository root. Use only the authenticated share URL remotely; the local listener binds to loopback.
