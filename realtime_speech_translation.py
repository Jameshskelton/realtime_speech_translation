# -----------------------------
# Imports
# -----------------------------

import os
import torch
import numpy as np
import gradio as gr

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from soprano import SopranoTTS
from scipy.io.wavfile import write
from pydub import AudioSegment


# -----------------------------
# Device and dtype configuration
# -----------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------
# Whisper ASR setup
# -----------------------------

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    chunk_length_s=5.0,
)


# -----------------------------
# Translation model setup
# -----------------------------

model_name_or_path = "tencent/HY-MT1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model_tr = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
)


# -----------------------------
# TTS setup
# -----------------------------

model_tts = SopranoTTS()


# -----------------------------
# Audio utilities
# -----------------------------

def convert_to_mono_pydub(input_file, output_file, output_format="wav"):
    """
    Converts a stereo or multi-channel audio file to mono using pydub.
    """
    audio = AudioSegment.from_file(input_file)
    mono_audio = audio.set_channels(1)
    mono_audio.export(output_file, format=output_format)
    print(f"Converted '{input_file}' to mono file '{output_file}'")


# -----------------------------
# ASR → Translation → TTS pipeline
# -----------------------------

def tts_translate(sample_audio):
    output_filename = "out1.wav"

    sample_rate, audio_array = sample_audio
    write(output_filename, sample_rate, audio_array)

    convert_to_mono_pydub("out1.wav", "out1.wav")

    result = pipe("out1.wav")

    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following segment into English, "
                "without additional explanation.\n\n"
                f"{result['text']}"
            ),
        }
    ]

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    outputs = model_tr.generate(
        tokenized_chat.to(model_tr.device),
        max_new_tokens=2048,
    )

    output_text = tokenizer.decode(outputs[0])

    innie = output_text.split("hy_place▁holder▁no▁8｜>")[1]
    clean_text = innie.split("<")[0]

    model_tts.infer(clean_text, "out.wav")

    return "out.wav", clean_text


# -----------------------------
# Gradio UI
# -----------------------------

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#F5F9FF",
        c100="#E5F2FF",
        c200="#B8D8FF",
        c300="#82BAFF",
        c400="#4A9BFF",
        c500="#0069FF",
        c600="#0061EB",
        c700="#0050C7",
        c800="#003DA5",
        c900="#031B4E",
        c950="#021533",
    ),
    neutral_hue=gr.themes.Color(
        c50="#F7FAFE",
        c100="#EEF4FA",
        c200="#DDE5EF",
        c300="#C1CDD9",
        c400="#94A3B8",
        c500="#64748B",
        c600="#475569",
        c700="#334155",
        c800="#1E293B",
        c900="#0F172A",
        c950="#020617",
    ),
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#F5F9FF",
    body_text_color="#031B4E",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E5F2FF",
    block_shadow="0 1px 4px rgba(0, 105, 255, 0.10)",
    block_title_text_color="#031B4E",
    button_primary_background_fill="#0069FF",
    button_primary_background_fill_hover="#0061EB",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#E5F2FF",
    button_secondary_background_fill_hover="#B8D8FF",
    button_secondary_text_color="#0050C7",
    input_background_fill="#FFFFFF",
    input_border_color="#DDE5EF",
)

css = """
.gradio-container { max-width: 960px !important; margin: 0 auto !important; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }
.app-header { text-align: center; margin-bottom: 0.5rem; }
.app-header h1 { color: #FFFFFF; margin-bottom: 0.25rem; }
.app-header p { color: #475569; font-size: 1.05rem; }
.app-footer { text-align: center; margin-top: 1.5rem; color: #94A3B8; font-size: 0.85rem; }
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.HTML(
        "<div class='app-header'>"
        "<h1>Real-Time Any-To-English Speech Translator</h1>"
        "<p>Record or upload audio in any language and get an English translation — text and speech.</p>"
        "</div>"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            with gr.Group():
                inp = gr.Audio(
                    label="Audio to Translate",
                    sources=["microphone", "upload"],
                )
            btn = gr.Button("Translate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            with gr.Group():
                out_audio = gr.Audio(label="Translated Audio", interactive=False)
                out_text = gr.Textbox(
                    label="Translated Text",
                    lines=6,
                    interactive=False,
                )

    btn.click(fn=tts_translate, inputs=inp, outputs=[out_audio, out_text])

    gr.HTML("<div class='app-footer'>Powered by Whisper &middot; HunyuanMT &middot; Soprano TTS</div>")

demo.launch()

