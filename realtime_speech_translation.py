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

with gr.Blocks() as demo:
    gr.Markdown("# Submit or record your audio for faster-than-speech translation!")

    with gr.Column():
        inp = gr.Audio(label="Input Audio to be Translated")

    with gr.Column():
        with gr.Row():
            out_audio = gr.Audio(label="Translated Audio")
        with gr.Row():
            out_text = gr.Textbox(label="Translated Text", lines=8)

    btn = gr.Button("Run")
    btn.click(fn=tts_translate, inputs=inp, outputs=[out_audio, out_text])

demo.launch()

