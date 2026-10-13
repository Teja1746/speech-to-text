"""Telangana Police STT Application.

A local, CPU-only speech-to-text web app (Telugu, Hindi, English) built with
Streamlit and faster-whisper. Audio is processed on this computer only.

Run with:  streamlit run app.py
"""

import csv
import io
import json
import os
import re
import tempfile
import time
import unicodedata

import jiwer
import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel

from config import (
    APP_TITLE,
    DEFAULT_MODEL,
    LANGUAGES,
    MODEL_DIR,
    MODELS,
    SUPPORTED_IN_APP,
)

# ---------------------------------------------------------------------------
# Page setup and theme (navy blue, white, soft grey, restrained gold accent)
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="centered")

st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; max-width: 900px; }
.app-header {
    background: linear-gradient(135deg, #0B2545 0%, #13315C 100%);
    color: #FFFFFF; padding: 1.4rem 1.6rem; border-radius: 10px;
    border-bottom: 4px solid #C9A227; margin-bottom: 1rem;
}
.app-header h1 { color: #FFFFFF; font-size: 1.8rem; margin: 0; padding: 0; }
.app-header p  { color: #D6DEEB; margin: 0.3rem 0 0 0; font-size: 1rem; }
h2, h3 { color: #0B2545; }
div.stButton > button[kind="primary"] {
    background-color: #0B2545; border: 1px solid #0B2545; color: #FFFFFF;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #13315C; border-color: #C9A227; color: #FFFFFF;
}
@media (max-width: 640px) {
    .app-header { padding: 1rem; }
    .app-header h1 { font-size: 1.3rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="app-header">
  <h1>{APP_TITLE}</h1>
  <p>Offline speech-to-text for Telugu, Hindi and English. Audio stays on this computer.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "🔒 **Privacy:** audio is processed locally on this computer. Uploaded audio is "
    "kept only in memory and in a temporary file that is deleted as soon as "
    "transcription ends. Transcript text is never written to logs by this app."
)
st.warning(
    "⚠️ **Human verification required.** Automated transcription can contain errors. "
    "Do not treat any transcript as verified evidence or use it for official purposes "
    "until a person has checked it against the original audio."
)

MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

# Session-state keys that belong to one transcription session.
SESSION_KEYS = (
    "transcript_text",
    "model_output",
    "audio_duration",
    "proc_time",
    "used_language",
    "used_model",
    "reference_text",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(repo_id: str) -> WhisperModel:
    """Load the model once and reuse it. First run downloads it into MODEL_DIR."""
    return WhisperModel(
        repo_id,
        device="cpu",
        compute_type="int8",  # small and fast on CPU
        cpu_threads=os.cpu_count() or 4,
        download_root=MODEL_DIR,
    )


def fmt_seconds(value: float) -> str:
    """Turn seconds into a friendly string such as '2m 5s'."""
    value = float(value)
    minutes, seconds = divmod(int(round(value)), 60)
    return f"{minutes}m {seconds}s" if minutes else f"{value:.1f}s"


def friendly_error(exc: Exception) -> str:
    """Convert a technical error into a message a beginner can act on."""
    text = str(exc).lower()
    if isinstance(exc, MemoryError) or any(
        word in text for word in ("out of memory", "bad_alloc", "cannot allocate", "memory")
    ):
        return (
            "The computer ran out of memory. Close other programs, or choose the "
            "'Whisper small (low-RAM)' or 'Whisper medium (lighter)' model, or split "
            "the recording into shorter parts."
        )
    if any(word in text for word in ("invalid data", "avcodec", "decode", "demux", "no audio")):
        return (
            "This file could not be read as audio. It may be corrupted or in an "
            "unsupported format. Please use a WAV, MP3, M4A or FLAC file."
        )
    if any(word in text for word in ("connection", "offline", "huggingface", "resolve", "timed out")):
        return (
            "The model could not be downloaded. Check the internet connection for "
            "the first run (the model is cached afterwards), or see the README "
            "troubleshooting section."
        )
    return f"Transcription failed ({type(exc).__name__}). Please try another file or see the README."


def transcribe_audio(model, audio_bytes, suffix, language_code, on_progress):
    """Transcribe bytes of audio. Returns (text, audio_duration_seconds).

    The audio is written to a temporary file (the decoder needs a file) and the
    file is always deleted afterwards, even if an error occurs.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # task="transcribe" keeps the original spoken language (no translation).
        segments, info = model.transcribe(
            tmp_path,
            language=language_code,
            task="transcribe",
            vad_filter=True,  # skips silence: faster, fewer made-up words
            beam_size=5,
            condition_on_previous_text=False,  # avoids repetition loops on long audio
        )
        total = info.duration or 0.0
        lines = []
        # 'segments' is a generator: audio is processed piece by piece, which
        # keeps memory use low even for long recordings.
        for segment in segments:
            piece = segment.text.strip()
            if piece:
                lines.append(piece)
            if total > 0:
                on_progress(min(segment.end / total, 1.0))
        return "\n".join(lines), total
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def normalize_text(text: str) -> str:
    """Make reference and prediction comparable: Unicode NFC, lowercase,
    punctuation (including the Hindi danda) replaced by spaces, extra spaces removed."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def copy_button(text: str) -> None:
    """A Copy Text button using a little JavaScript (works inside Streamlit)."""
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        """
<button id="copybtn" style="width:100%;height:42px;border-radius:8px;border:1px solid #0B2545;
  background:#FFFFFF;color:#0B2545;font-size:15px;cursor:pointer;">📋 Copy Text</button>
<script>
const text = __PAYLOAD__;
const btn = document.getElementById('copybtn');
function done() {
  btn.innerText = '✅ Copied';
  setTimeout(() => { btn.innerText = '📋 Copy Text'; }, 1500);
}
btn.onclick = async () => {
  try {
    await navigator.clipboard.writeText(text);
    done();
  } catch (err) {
    const area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    document.body.removeChild(area);
    done();
  }
};
</script>
""".replace("__PAYLOAD__", payload),
        height=50,
    )


def reset_all() -> None:
    """Clear results and the upload/record widgets (runs before the page redraws)."""
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1


# ---------------------------------------------------------------------------
# 1. Language and model selection
# ---------------------------------------------------------------------------
run_id = st.session_state.setdefault("run_id", 0)

st.subheader("1. Choose language")
col_language, col_model = st.columns(2)
with col_language:
    language_label = st.selectbox("Spoken language", list(LANGUAGES.keys()))
with col_model:
    model_label = st.selectbox(
        "Model",
        list(MODELS.keys()),
        index=list(MODELS.keys()).index(DEFAULT_MODEL),
        help="The recommended model is the most accurate. Use a lighter one if memory is low.",
    )
language_code = LANGUAGES[language_label]
model_info = MODELS[model_label]

with st.expander("ℹ️ Selected model and supported languages"):
    st.markdown(
        f"""
- **Model:** {model_label}
- **Hugging Face page:** [{model_info['repo']}]({model_info['url']})
- **License:** {model_info['license']}
- **Download size:** {model_info['download']}
- **RAM needed:** {model_info['ram']}
- **Languages in this app:** {SUPPORTED_IN_APP} (the underlying model supports about 100 languages)
- **Runs on:** CPU only, no GPU and no paid API key
- **Limitations:** accuracy varies with accent, background noise and recording quality, and is
  generally lower for Telugu than for English or Hindi. Names, numbers and mixed-language
  speech are common error sources. Always check results with the WER/CER tool and a human reviewer.
"""
    )

# ---------------------------------------------------------------------------
# 2. Audio input
# ---------------------------------------------------------------------------
st.subheader("2. Add audio")
tab_upload, tab_record = st.tabs(["📁 Upload audio", "🎙️ Record with microphone"])

uploaded = None
recorded = None
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a WAV, MP3, M4A or FLAC file",
        type=["wav", "mp3", "m4a", "flac"],
        key=f"upload_{run_id}",
    )
with tab_record:
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Click the microphone to record", key=f"mic_{run_id}")
        st.caption(
            "Browsers allow microphone access only on localhost or HTTPS pages. "
            "If a file is uploaded as well, the uploaded file is used."
        )
    else:
        st.info("Microphone recording needs Streamlit 1.39 or newer. Run: pip install -U streamlit")

audio_bytes = None
suffix = ".wav"
mime = "audio/wav"
if uploaded is not None:
    audio_bytes = uploaded.getvalue()
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
    mime = MIME_TYPES.get(suffix, "audio/wav")
elif recorded is not None:
    audio_bytes = recorded.getvalue()

if audio_bytes:
    st.audio(audio_bytes, format=mime)  # playback controls

# ---------------------------------------------------------------------------
# 3. Transcribe
# ---------------------------------------------------------------------------
st.subheader("3. Transcribe")
if st.button(
    "▶ Transcribe",
    type="primary",
    use_container_width=True,
    disabled=audio_bytes is None,
):
    if not audio_bytes:
        st.error("The audio file is empty. Please choose a valid recording.")
    else:
        try:
            with st.spinner(
                "Loading the model. The first run downloads it and may take several minutes..."
            ):
                model = load_model(model_info["repo"])

            progress = st.progress(0.0, text="Starting transcription...")

            def on_progress(fraction: float) -> None:
                progress.progress(fraction, text=f"Transcribing... {int(fraction * 100)}%")

            started = time.perf_counter()
            text, duration = transcribe_audio(
                model, audio_bytes, suffix, language_code, on_progress
            )
            elapsed = time.perf_counter() - started
            progress.empty()

            if not text.strip():
                st.warning(
                    "No speech was detected. Check that the recording has audible "
                    "speech and that the correct language is selected."
                )
            else:
                st.session_state["transcript_text"] = text
                st.session_state["model_output"] = text
                st.session_state["audio_duration"] = duration
                st.session_state["proc_time"] = elapsed
                st.session_state["used_language"] = language_label
                st.session_state["used_model"] = model_label
        except Exception as exc:  # show a friendly message instead of crashing
            st.error(friendly_error(exc))
            st.exception(exc)

# ---------------------------------------------------------------------------
# 4. Result: edit, copy, download
# ---------------------------------------------------------------------------
if "model_output" in st.session_state:
    st.subheader("4. Transcription")
    m1, m2, m3 = st.columns(3)
    m1.metric("Audio length", fmt_seconds(st.session_state["audio_duration"]))
    m2.metric("Processing time", fmt_seconds(st.session_state["proc_time"]))
    m3.metric("Language", st.session_state["used_language"].split(" ")[0])

    st.text_area(
        "Transcription (you can edit this text)",
        key="transcript_text",
        height=300,
    )
    final_text = st.session_state["transcript_text"]

    # CSV: one row per line of text, UTF-8 with BOM so Excel shows Telugu/Hindi correctly.
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["line_no", "text"])
    for number, line in enumerate([ln for ln in final_text.splitlines() if ln.strip()], start=1):
        writer.writerow([number, line])

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        copy_button(final_text)
    with b2:
        st.download_button(
            "⬇ TXT",
            data=final_text.encode("utf-8"),
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with b3:
        st.download_button(
            "⬇ CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="transcription.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b4:
        st.button("🗑 Reset", on_click=reset_all, use_container_width=True)

    # -----------------------------------------------------------------------
    # 5. Optional WER / CER evaluation
    # -----------------------------------------------------------------------
    with st.expander("📊 Evaluate accuracy (WER and CER)"):
        st.write(
            "Type or paste the **correct** transcript below. The app compares it with the "
            "text and shows the Word Error Rate (WER) and Character Error Rate (CER). "
            "**Lower is better**; 0% means a perfect match."
        )
        reference = st.text_area("Correct reference transcript", key="reference_text", height=150)
        compare_with = st.radio(
            "Compare the reference with",
            ["Original model output", "My edited text"],
            horizontal=True,
        )
        if st.button("Calculate WER and CER"):
            hypothesis_raw = (
                st.session_state["model_output"]
                if compare_with == "Original model output"
                else final_text
            )
            ref_norm = normalize_text(reference)
            hyp_norm = normalize_text(hypothesis_raw)
            if not ref_norm:
                st.warning("Please enter the correct reference transcript first.")
            elif not hyp_norm:
                st.warning("The text to compare is empty.")
            else:
                wer_value = jiwer.wer(ref_norm, hyp_norm) * 100
                cer_value = jiwer.cer(ref_norm, hyp_norm) * 100
                r1, r2 = st.columns(2)
                r1.metric("Word Error Rate (WER)", f"{wer_value:.1f}%")
                r2.metric("Character Error Rate (CER)", f"{cer_value:.1f}%")
                st.caption(
                    "Both texts were normalized the same way before comparing "
                    "(Unicode normalization, lowercase, punctuation removed, extra spaces removed). "
                    "Values above 100% are possible when the output has many extra words."
                )

st.divider()
st.caption(
    f"{APP_TITLE} is an independent tool. It is not an official product and has no "
    "government approval. Transcripts require human verification."
)
