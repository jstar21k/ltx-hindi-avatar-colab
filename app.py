import asyncio
import gc
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import edge_tts
import gradio as gr
import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers.utils import export_to_video


MODEL_ID = "Lightricks/LTX-Video-0.9.7-distilled"
OUTPUT_DIR = Path("/content/ltx_avatar_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPE = None


DEFAULT_PROMPT = (
    "A realistic close-up portrait video of an adult Indian woman avatar speaking warmly to the camera. "
    "She keeps natural eye contact, blinks softly, makes subtle head movements, and smiles gently. "
    "The camera is stable, portrait composition, shallow depth of field, clean studio lighting, "
    "realistic skin texture, natural facial motion, no exaggerated expression."
)

DEFAULT_NEGATIVE = (
    "worst quality, low quality, blurry, distorted face, deformed mouth, extra teeth, bad eyes, "
    "crossed eyes, jitter, flicker, warped face, unnatural skin, plastic skin, duplicate face, "
    "text, watermark"
)

DEFAULT_HINDI = (
    "नमस्ते, मैं आपकी डिजिटल अवतार हूं। आज मैं आपसे हिंदी में बात कर रही हूं। "
    "यह वीडियो एक इमेज और प्रॉम्प्ट से बनाया गया है।"
)


DEFAULT_HINDI = (
    "\u0928\u092e\u0938\u094d\u0924\u0947, \u092e\u0948\u0902 \u0906\u092a\u0915\u0940 "
    "\u0921\u093f\u091c\u093f\u091f\u0932 \u0905\u0935\u0924\u093e\u0930 \u0939\u0942\u0902\u0964 "
    "\u0906\u091c \u092e\u0948\u0902 \u0906\u092a\u0938\u0947 \u0939\u093f\u0902\u0926\u0940 "
    "\u092e\u0947\u0902 \u092c\u093e\u0924 \u0915\u0930 \u0930\u0939\u0940 \u0939\u0942\u0902\u0964 "
    "\u092f\u0939 \u0935\u0940\u0921\u093f\u092f\u094b \u090f\u0915 \u0907\u092e\u0947\u091c "
    "\u0914\u0930 \u092a\u094d\u0930\u0949\u092e\u094d\u092a\u094d\u091f \u0938\u0947 "
    "\u092c\u0928\u093e\u092f\u093e \u0917\u092f\u093e \u0939\u0948\u0964"
)


def ensure_gpu() -> str:
    if not torch.cuda.is_available():
        raise gr.Error("GPU is not enabled. In Colab use Runtime > Change runtime type > T4 GPU.")
    return torch.cuda.get_device_name(0)


def get_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE

    ensure_gpu()
    major, _minor = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    PIPE = LTXConditionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    PIPE.enable_model_cpu_offload()
    PIPE.vae.enable_tiling()
    return PIPE


def dimensions_for(style: str) -> tuple[int, int]:
    if style == "Vertical 9:16":
        return 480, 704
    if style == "Wide 16:9":
        return 704, 480
    return 576, 576


def frame_count(seconds: int, fps: int) -> int:
    frames = int(seconds * fps)
    return ((frames - 1) // 8) * 8 + 1


def save_hindi_voice(text: str, voice: str, output_path: Path) -> None:
    async def _save():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_save())
    finally:
        loop.close()


def merge_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise gr.Error("FFmpeg failed while merging audio and video:\n" + result.stderr[-1200:])


def generate_avatar(image, prompt, hindi_text, negative_prompt, style, seconds, fps, seed, voice, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("Upload an avatar image first.")
    if not prompt.strip():
        raise gr.Error("Write a video prompt.")
    if not hindi_text.strip():
        raise gr.Error("Write Hindi speech text.")

    gpu_name = ensure_gpu()
    width, height = dimensions_for(style)
    num_frames = frame_count(seconds, fps)
    job_id = uuid.uuid4().hex[:10]
    raw_video_path = OUTPUT_DIR / f"{job_id}_raw.mp4"
    audio_path = OUTPUT_DIR / f"{job_id}_hindi.mp3"
    final_path = OUTPUT_DIR / f"{job_id}_final.mp4"

    progress(0.05, desc=f"Loading LTX on {gpu_name}")
    pipe = get_pipe()
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    condition = LTXVideoCondition(image=image.convert("RGB"), frame_index=0)

    progress(0.20, desc="Generating avatar motion")
    started = time.time()
    frames = pipe(
        conditions=[condition],
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE,
        width=width,
        height=height,
        num_frames=num_frames,
        timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        image_cond_noise_scale=0.025,
        guidance_scale=1.0,
        guidance_rescale=0.7,
        generator=generator,
        output_type="pil",
    ).frames[0]

    progress(0.72, desc="Encoding video")
    export_to_video(frames, str(raw_video_path), fps=fps)

    progress(0.82, desc="Generating Hindi voice")
    save_hindi_voice(hindi_text.strip(), voice, audio_path)

    progress(0.92, desc="Merging audio")
    merge_audio(raw_video_path, audio_path, final_path)

    del frames
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = round(time.time() - started, 1)
    status = f"Done: {seconds}s, {width}x{height}, {fps} fps, seed {seed}. Render time: {elapsed}s."
    return str(final_path), str(final_path), status


def build_ui():
    css = """
    .gradio-container { max-width: 1180px !important; margin: auto !important; }
    .hero {
        padding: 22px 0 8px 0;
        border-bottom: 1px solid rgba(255,255,255,.12);
        margin-bottom: 18px;
    }
    .hero h1 { font-size: 32px; margin: 0 0 8px 0; letter-spacing: 0; }
    .hero p { font-size: 15px; margin: 0; opacity: .78; max-width: 780px; }
    #generate_btn button { font-weight: 700; height: 48px; }
    #status_box textarea { font-family: ui-monospace, Consolas, monospace; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), title="LTX Hindi Avatar Studio") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>LTX Hindi Avatar Studio</h1>
              <p>Upload an avatar image, write a motion prompt and Hindi dialogue, then generate a short video with voice.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                image = gr.Image(label="Avatar image", type="pil", height=330)
                prompt = gr.Textbox(label="Video prompt", value=DEFAULT_PROMPT, lines=5)
                hindi_text = gr.Textbox(label="Hindi dialogue", value=DEFAULT_HINDI, lines=4)

                with gr.Accordion("Advanced settings", open=False):
                    negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE, lines=3)
                    with gr.Row():
                        style = gr.Dropdown(["Vertical 9:16", "Wide 16:9", "Square"], value="Vertical 9:16", label="Format")
                        voice = gr.Dropdown(
                            ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
                            value="hi-IN-SwaraNeural",
                            label="Hindi voice",
                        )
                    with gr.Row():
                        seconds = gr.Slider(5, 15, value=5, step=1, label="Duration")
                        fps = gr.Slider(16, 24, value=24, step=1, label="FPS")
                        seed = gr.Number(value=12345, precision=0, label="Seed")

                generate_btn = gr.Button("Generate Video", variant="primary", elem_id="generate_btn")

            with gr.Column(scale=6, min_width=420):
                video = gr.Video(label="Preview")
                download = gr.File(label="Download MP4")
                status = gr.Textbox(label="Status", lines=3, elem_id="status_box")

        generate_btn.click(
            fn=generate_avatar,
            inputs=[image, prompt, hindi_text, negative_prompt, style, seconds, fps, seed, voice],
            outputs=[video, download, status],
        )

        gr.Markdown(
            "Use fictional or consenting adult avatars only. Avoid deceptive impersonation and non-consensual content."
        )

    return demo


def launch():
    demo = build_ui()
    demo.queue(max_size=8).launch(share=True, debug=False, show_error=True)


if __name__ == "__main__":
    launch()
