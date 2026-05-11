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

    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline

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
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video

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

    progress(0.05, desc=f"Loading & Downloading LTX Model on {gpu_name} (Takes 2-5 mins first time)")
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
    :root {
        --studio-bg: #111216;
        --studio-panel: #1b1c21;
        --studio-panel-2: #15161a;
        --studio-border: rgba(255,255,255,.12);
        --studio-text: #f4f7fb;
        --studio-muted: #9aa3b2;
        --studio-accent: #13c8ff;
    }
    body, .gradio-container {
        background:
            radial-gradient(circle at 50% -10%, rgba(24, 190, 210, .16), transparent 34%),
            radial-gradient(circle at 65% 12%, rgba(155, 75, 220, .14), transparent 28%),
            var(--studio-bg) !important;
    }
    .gradio-container {
        max-width: 1240px !important;
        margin: auto !important;
        color: var(--studio-text) !important;
    }
    .studio-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 18px 2px 10px;
    }
    .studio-brand {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        background: rgba(255,255,255,.045);
        border: 1px solid var(--studio-border);
        border-radius: 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,.28);
        font-weight: 700;
    }
    .studio-mark {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0ce2ff, #2f7cff);
        display: grid;
        place-items: center;
        color: white;
        font-weight: 900;
    }
    .studio-badge {
        border: 1px solid var(--studio-border);
        color: var(--studio-muted);
        padding: 9px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.035);
        font-size: 13px;
    }
    .studio-hero {
        text-align: center;
        padding: 24px 0 28px;
    }
    .studio-hero h1 {
        font-size: clamp(34px, 5vw, 56px);
        line-height: 1.02;
        margin: 0 0 12px;
        letter-spacing: 0;
        color: var(--studio-text);
    }
    .studio-hero p {
        margin: 0 auto;
        color: var(--studio-muted);
        font-size: 16px;
        max-width: 760px;
    }
    #creator_shell {
        max-width: 980px;
        margin: 0 auto 22px;
        padding: 20px;
        border-radius: 34px;
        border: 1px solid var(--studio-border);
        background: rgba(28,29,34,.82);
        box-shadow: 0 26px 100px rgba(0,0,0,.34);
    }
    #mode_bar {
        padding: 6px;
        border-radius: 22px;
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.06);
        margin-bottom: 18px;
    }
    #mode_bar .tab-nav {
        background: transparent !important;
        border: 0 !important;
    }
    #mode_bar button {
        border-radius: 18px !important;
        min-height: 44px;
        font-weight: 700;
    }
    #script_box textarea, #prompt_box textarea {
        min-height: 238px !important;
        background: var(--studio-panel-2) !important;
        border-radius: 22px !important;
        border: 1px solid rgba(255,255,255,.08) !important;
        color: var(--studio-text) !important;
        font-size: 15px !important;
    }
    #prompt_box textarea { min-height: 112px !important; }
    #image_box {
        min-height: 334px;
        border-radius: 24px !important;
        border: 1px dashed rgba(255,255,255,.18) !important;
        background: rgba(255,255,255,.025) !important;
    }
    #controls_row {
        align-items: end;
        margin-top: 14px;
    }
    #generate_btn button {
        height: 52px;
        border-radius: 999px !important;
        background: var(--studio-accent) !important;
        color: #061016 !important;
        font-weight: 800;
        border: 0 !important;
        box-shadow: 0 16px 40px rgba(19,200,255,.24);
    }
    #output_shell {
        max-width: 980px;
        margin: 0 auto;
        padding: 18px;
        border-radius: 28px;
        border: 1px solid var(--studio-border);
        background: rgba(18,19,23,.72);
    }
    #status_box textarea {
        font-family: ui-monospace, Consolas, monospace;
        background: rgba(255,255,255,.035) !important;
        border-radius: 16px !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), title="LTX Hindi Avatar Studio") as demo:
        gr.HTML(
            """
            <div class="studio-topbar">
              <div class="studio-brand"><div class="studio-mark">A</div><span>Avatar videos</span></div>
              <div class="studio-badge">Photo to video with Hindi voice</div>
            </div>
            <div class="studio-hero">
              <h1>One image. One prompt. A talking avatar video.</h1>
              <p>Upload a face image, write the Hindi script and motion prompt, then generate a short video with voice.</p>
            </div>
            """
        )

        with gr.Group(elem_id="creator_shell"):
            with gr.Tabs(elem_id="mode_bar"):
                with gr.Tab("Photo to video"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5, min_width=340):
                            hindi_text = gr.Textbox(
                                label="Hindi script",
                                value=DEFAULT_HINDI,
                                placeholder="Type or paste Hindi dialogue here...",
                                lines=9,
                                elem_id="script_box",
                            )
                        with gr.Column(scale=6, min_width=360):
                            image = gr.Image(
                                label="Upload photo",
                                type="pil",
                                height=334,
                                elem_id="image_box",
                            )

                    prompt = gr.Textbox(
                        label="Motion prompt",
                        value=DEFAULT_PROMPT,
                        placeholder="Describe the avatar movement, camera, expression, lighting...",
                        lines=3,
                        elem_id="prompt_box",
                    )

                    with gr.Row(elem_id="controls_row"):
                        voice = gr.Dropdown(
                            ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
                            value="hi-IN-SwaraNeural",
                            label="Voice",
                            scale=2,
                        )
                        style = gr.Dropdown(
                            ["Vertical 9:16", "Wide 16:9", "Square"],
                            value="Vertical 9:16",
                            label="Format",
                            scale=2,
                        )
                        seconds = gr.Slider(5, 15, value=5, step=1, label="Seconds", scale=3)
                        fps = gr.Slider(16, 24, value=24, step=1, label="FPS", scale=2)
                        generate_btn = gr.Button("Generate", variant="primary", scale=2, elem_id="generate_btn")

                    with gr.Accordion("Advanced", open=False):
                        seed = gr.Number(value=12345, precision=0, label="Seed")
                        negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE, lines=3)

                with gr.Tab("Script to video"):
                    gr.Markdown("Script-to-video mode is not enabled yet. Use Photo to video for this notebook.")

        with gr.Group(elem_id="output_shell"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    video = gr.Video(label="Preview")
                with gr.Column(scale=4):
                    download = gr.File(label="Download MP4")
                    status = gr.Textbox(label="Status", lines=5, elem_id="status_box")

        generate_btn.click(
            fn=generate_avatar,
            inputs=[image, prompt, hindi_text, negative_prompt, style, seconds, fps, seed, voice],
            outputs=[video, download, status],
            show_progress="minimal"
        )

        gr.Markdown("Use fictional or consenting adult avatars only. Avoid deceptive impersonation and non-consensual content.")

    return demo


def launch():
    demo = build_ui()
    demo.queue(max_size=8).launch(share=True, debug=False, show_error=True, inline=True, height=1200)


if __name__ == "__main__":
    launch()
