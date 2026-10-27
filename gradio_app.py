import os
import json
import math
import tempfile
import threading
import time
import warnings
from pathlib import Path
from uuid import uuid4

CKPT_DIR = "/content/models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR = "/content/models/wav2vec2-base-960h"
MODEL_TYPE = "lite"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
RESULTS_DIR = Path(__file__).absolute().parent / "gradio_results"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "soulx-flashhead-uploads"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = str(UPLOAD_DIR)


def read_credentials():
    username = os.environ.get("APP_USERNAME", "weblogic").strip()
    password = os.environ.get("APP_PASSWORD", "Aug-2015")
    if not username or not password:
        raise SystemExit("Set APP_USERNAME and APP_PASSWORD before starting the app.")
    return username, password


# Validate credentials before loading the UI or GPU dependencies.
username, password = read_credentials()

import gradio as gr
import torch
import numpy as np
import imageio
import librosa
import subprocess
from collections import deque
from PIL import Image

from flash_head.inference import (
    get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline,
)

pipeline = None


def cleanup_results():
    # Never traverse directories or follow links outside the fixed results folder.
    if RESULTS_DIR.is_symlink():
        raise ValueError("Invalid results directory")
    RESULTS_DIR.mkdir(exist_ok=True)
    cutoff = time.time() - 3600
    for path in RESULTS_DIR.iterdir():
        if not path.is_symlink() and path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def cleanup_periodically():
    while True:
        try:
            cleanup_results()
        except Exception:
            pass  # Do not expose paths or exception details.
        time.sleep(300)


def validate_uploads(cond_image, audio_path):
    try:
        paths = [Path(cond_image), Path(audio_path)]
        for path in paths:
            # Accept only uploaded cache files, never arbitrary server paths.
            if path.is_symlink() or not path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
                raise ValueError("Invalid upload")
            if not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
                raise ValueError("Invalid upload")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(paths[0]) as image:
                if not (0 < image.width <= 4096 and 0 < image.height <= 4096):
                    raise ValueError("Invalid image dimensions")
                image.verify()
            with Image.open(paths[0]) as image:
                image.load()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type,duration:format=duration",
             "-of", "json", str(paths[1].resolve())],
            capture_output=True, text=True, check=True, timeout=15,
        )
        metadata = json.loads(probe.stdout)
        streams = metadata.get("streams", [])
        if not streams or streams[0].get("codec_type") != "audio":
            raise ValueError("No audio stream")
        duration = float(metadata.get("format", {}).get("duration", streams[0].get("duration", "nan")))
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError("Invalid audio duration")
    except Exception:
        raise gr.Error(
            "Upload a valid image up to 4096 x 4096 pixels and nonempty audio up to "
            "30 seconds. Each file must be 12 MB or smaller."
        ) from None


def save_video_to_file(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace('.mp4', '_temp.mp4')
    try:
        with imageio.get_writer(
            temp_video_path, format='mp4', mode='I', fps=fps,
            codec='h264', ffmpeg_params=['-bf', '0'],
        ) as writer:
            for frames in frames_list:
                for frame in frames.numpy().astype(np.uint8):
                    writer.append_data(frame)
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', temp_video_path, '-i', audio_path,
             '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y'],
            check=True, capture_output=True, timeout=120,
        )
    finally:
        Path(temp_video_path).unlink(missing_ok=True)
    return video_path


def run_inference(cond_image, audio_path, seed, use_face_crop, progress=gr.Progress()):
    global pipeline
    try:
        cleanup_results()
        validate_uploads(cond_image, audio_path)
        return generate_video(cond_image, audio_path, seed, use_face_crop, progress)
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Video generation failed. Please check your media and try again.") from None


def generate_video(cond_image, audio_path, seed, use_face_crop, progress):
    global pipeline
    # Decode and check the actual sample count before loading the GPU pipeline.
    infer_params = get_infer_params()
    sample_rate = infer_params['sample_rate']
    human_speech_array_all, _ = librosa.load(
        audio_path, sr=sample_rate, mono=True, duration=30 + 1 / sample_rate,
    )
    if not 0 < len(human_speech_array_all) <= 30 * sample_rate:
        raise gr.Error("Audio must be nonempty and no longer than 30 seconds.")
    if not np.isfinite(human_speech_array_all).all():
        raise gr.Error("Please upload a valid audio file.")
    if pipeline is None:
        progress(0, desc="Loading model...")
        pipeline = get_pipeline(
            world_size=1, ckpt_dir=CKPT_DIR, model_type=MODEL_TYPE,
            wav2vec_dir=WAV2VEC_DIR,
        )
    base_seed = int(seed) if seed >= 0 else 9999
    progress(0.1, desc="Preparing media...")
    get_base_data(
        pipeline, cond_image_path_or_dir=cond_image, base_seed=base_seed,
        use_face_crop=use_face_crop,
    )
    infer_params = get_infer_params()
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    generated_list = []

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    # pad audio with silence to avoid truncating the last chunk
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate([human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)])

    # split audio embedding into chunks: 28, 28, 28, 28, ...
    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

    total_chunks = len(human_speech_array_slices)
    for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
        progress(0.2 + 0.7 * (chunk_idx / total_chunks), desc=f"Generating chunk {chunk_idx+1}/{total_chunks}")
        
        # streaming encode audio chunks
        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

        torch.cuda.synchronize()

        # inference
        video = run_pipeline(pipeline, audio_embedding)
        video = video[motion_frames_num:]

        torch.cuda.synchronize()

        generated_list.append(video.cpu())

    progress(0.95, desc="Saving video...")
    save_path = str(RESULTS_DIR / f"res_{uuid4().hex}.mp4")
    return save_video_to_file(generated_list, save_path, audio_path, fps=tgt_fps)


with gr.Blocks(
    title="SoulX-FlashHead Video Generator", theme=gr.themes.Soft(),
    analytics_enabled=False, delete_cache=(1800, 1800),
) as app:
    gr.Markdown("# ⚡ SoulX-FlashHead Video Generator")
    gr.Markdown("Upload an image and an audio file to generate a talking head video.")
    gr.Markdown(
        "Upload only media you own or have permission to use. Outputs are deleted automatically."
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎬 Generation Inputs")
            # File components preserve original bytes for size and format validation.
            cond_image_input = gr.File(
                label="Condition Image (up to 4096 x 4096, 12 MB)",
                type="filepath", file_types=["image"],
            )
            audio_path_input = gr.File(
                label="Audio Input (up to 30 seconds, 12 MB)",
                type="filepath", file_types=["audio"],
            )
            with gr.Accordion("⚙️ Inference Settings", open=False):
                use_face_crop_input = gr.Checkbox(label="Use Face Crop", value=False)
                seed_input = gr.Number(label="Random Seed", value=9999, precision=0)
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Output Video")
            video_output = gr.Video(label="Generated Video", height=500)
    generate_btn.click(
        fn=run_inference,
        inputs=[cond_image_input, audio_path_input, seed_input, use_face_crop_input],
        outputs=video_output, concurrency_limit=1, api_name=False,
    )

# Gradio 5.50.0 has no run-history feature or launch argument. Keep the explicit
# privacy setting without upgrading into a Transformers/huggingface-hub conflict.
_launch_without_history = app.launch


def launch_without_history(*, run_history=False, **kwargs):
    if run_history is not False:
        raise ValueError("Run history must remain disabled.")
    return _launch_without_history(**kwargs)


app.launch = launch_without_history

if __name__ == "__main__":
    threading.Thread(target=cleanup_periodically, daemon=True).start()
    app.queue(default_concurrency_limit=1, max_size=2)
    app.launch(
        server_name="127.0.0.1",
        share=True,
        auth=(username, password),
        auth_message="Private POC — authorized testers only.",
        max_file_size="12mb",
        max_threads=1,
        show_error=False,
        run_history=False,
        enable_monitoring=False,
        strict_cors=True,
        blocked_paths=[
            "/content/models",
            "/root/.cache",
            "/content/.cache",
        ],
    )
