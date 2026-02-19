# -----------------------------
# Imports
# -----------------------------

import os
import tempfile
import threading
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
# Configuration
# -----------------------------

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
TRANSLATION_MODEL_ID = "tencent/HY-MT1.5-1.8B"
ASR_CHUNK_LENGTH_S = 5.0
TRANSLATION_MAX_NEW_TOKENS = 512

TARGET_LANGUAGES = [
    "English", "French", "German", "Spanish", "Portuguese",
    "Italian", "Chinese", "Japanese", "Korean", "Arabic",
    "Russian", "Hindi",
]

# Maps target language display names to Whisper language codes for
# skip-translation detection.
LANGUAGE_CODES = {
    "English": "en", "French": "fr", "German": "de", "Spanish": "es",
    "Portuguese": "pt", "Italian": "it", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Arabic": "ar", "Russian": "ru", "Hindi": "hi",
}


# -----------------------------
# Device and dtype configuration
# -----------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------
# Lazy model loading
# -----------------------------

_models = {}
_models_lock = threading.Lock()


def _load_models():
    """Load all models on first call; subsequent calls are no-ops."""
    with _models_lock:
        if _models:
            return

        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        asr_model.to(device)

        processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)

        _models["asr"] = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
            chunk_length_s=ASR_CHUNK_LENGTH_S,
        )

        _models["tokenizer"] = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
        _models["translation"] = AutoModelForCausalLM.from_pretrained(
            TRANSLATION_MODEL_ID,
            device_map="auto",
        )

        _models["tts"] = SopranoTTS()


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

def tts_translate(sample_audio, target_language="English"):
    if sample_audio is None:
        raise gr.Error("No audio provided. Please record or upload audio first.")

    _load_models()

    tmp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(tmp_dir, "raw.wav")
    mono_path = os.path.join(tmp_dir, "mono.wav")
    output_path = os.path.join(tmp_dir, "out.wav")

    try:
        sample_rate, audio_array = sample_audio
        write(raw_path, sample_rate, audio_array)
        convert_to_mono_pydub(raw_path, mono_path)
    except Exception as e:
        raise gr.Error(f"Failed to process input audio: {e}")

    try:
        result = _models["asr"](mono_path)
    except Exception as e:
        raise gr.Error(f"Speech recognition failed: {e}")

    transcribed_text = result.get("text", "").strip()
    if not transcribed_text:
        raise gr.Error("Could not recognize any speech in the audio. Please try again with clearer audio.")

    # Whisper may return a language code (e.g. "en", "fr") in the result.
    # Degrade gracefully if it isn't present.
    detected_language_code = result.get("language")
    display_language = detected_language_code.upper() if detected_language_code else "Unknown"

    target_code = LANGUAGE_CODES.get(target_language, "").lower()
    already_in_target = bool(
        detected_language_code
        and target_code
        and detected_language_code.lower() == target_code
    )

    if already_in_target:
        translated_text = transcribed_text
    else:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Translate the following segment into {target_language}, "
                        "without additional explanation.\n\n"
                        f"{transcribed_text}"
                    ),
                }
            ]

            tokenized_chat = _models["tokenizer"].apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )

            outputs = _models["translation"].generate(
                tokenized_chat.to(_models["translation"].device),
                max_new_tokens=TRANSLATION_MAX_NEW_TOKENS,
            )

            output_text = _models["tokenizer"].decode(outputs[0])
        except Exception as e:
            raise gr.Error(f"Translation failed: {e}")

        parts = output_text.split("hy_place▁holder▁no▁8｜>")
        if len(parts) < 2:
            raise gr.Error("Translation model returned unexpected output. Please try again.")
        translated_text = parts[1].split("<")[0].strip()

        if not translated_text:
            raise gr.Error("Translation produced empty text. Please try again with different audio.")

    try:
        _models["tts"].infer(translated_text, output_path)
    except Exception as e:
        raise gr.Error(f"Text-to-speech failed: {e}")

    return output_path, transcribed_text, display_language, translated_text


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
        "<p>Record or upload audio in any language and get a translation — text and speech.</p>"
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
            target_lang = gr.Dropdown(
                choices=TARGET_LANGUAGES,
                value="English",
                label="Target Language",
            )
            btn = gr.Button("Translate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            with gr.Group():
                out_source_lang = gr.Textbox(
                    label="Detected Source Language",
                    interactive=False,
                    lines=1,
                )
                out_transcription = gr.Textbox(
                    label="Original Transcription",
                    lines=3,
                    interactive=False,
                )
                out_audio = gr.Audio(label="Translated Audio", interactive=False)
                out_text = gr.Textbox(
                    label="Translated Text",
                    lines=4,
                    interactive=False,
                )

    btn.click(
        fn=tts_translate,
        inputs=[inp, target_lang],
        outputs=[out_audio, out_transcription, out_source_lang, out_text],
        show_progress="full",
    )

    gr.HTML("<div class='app-footer'>Powered by Whisper &middot; HunyuanMT &middot; Soprano TTS</div>")

demo.launch(server_name=os.environ.get("SERVER_HOST", "127.0.0.1"))
