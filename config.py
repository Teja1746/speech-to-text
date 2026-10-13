"""Shared settings for the Telangana Police STT Application.

Both app.py and download_model.py import from this file, so the model list
lives in exactly one place.
"""

APP_TITLE = "Telangana Police STT Application"

# Folder (inside the project) where downloaded models are cached.
MODEL_DIR = "models"

# Spoken-language menu: label shown to the user -> Whisper language code.
LANGUAGES = {
    "Telugu (తెలుగు)": "te",
    "Hindi (हिन्दी)": "hi",
    "English": "en",
}

# Models the app can use. All are Whisper models converted to the
# CTranslate2 format that faster-whisper runs efficiently on CPU.
# Sizes marked "about" or "estimate" are NOT taken from the model page;
# check them on your own machine (see README, "RAM and speed").
MODELS = {
    "Whisper large-v3-turbo (recommended)": {
        "key": "turbo",
        "repo": "/home/tejas/tgstt/models/faster-whisper-large-v3-turbo-ct2",
        "url": "https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2",
        "download": "about 1.6 GB (model.bin is listed as 1.62 GB)",
        "ram": "about 3-4 GB with int8 (estimate, measure on your machine)",
        "license": "MIT",
    },
    "Whisper medium (lighter)": {
        "key": "medium",
        "repo": "Systran/faster-whisper-medium",
        "url": "https://huggingface.co/Systran/faster-whisper-medium",
        "download": "about 1.5 GB (approximate)",
        "ram": "about 2-3 GB with int8 (estimate, measure on your machine)",
        "license": "MIT",
    },
    "Whisper small (low-RAM)": {
        "key": "small",
        "repo": "Systran/faster-whisper-small",
        "url": "https://huggingface.co/Systran/faster-whisper-small",
        "download": "about 0.5 GB (approximate)",
        "ram": "about 1 GB with int8 (estimate, measure on your machine)",
        "license": "MIT",
    },
}

DEFAULT_MODEL = "Whisper large-v3-turbo (recommended)"

# Languages the underlying Whisper models can transcribe: roughly 100.
# This app only exposes the three the project needs.
SUPPORTED_IN_APP = "Telugu, Hindi, English"
