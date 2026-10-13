"""Download and cache a speech model before running the app.

Usage:
    python download_model.py          # recommended model (large-v3-turbo)
    python download_model.py medium   # lighter model
    python download_model.py small    # lowest-RAM model

The files are saved inside the "models" folder of this project, so the app
never needs to download them again.
"""

import sys

from faster_whisper import WhisperModel

from config import DEFAULT_MODEL, MODEL_DIR, MODELS


def main() -> None:
    wanted = sys.argv[1].lower() if len(sys.argv) > 1 else MODELS[DEFAULT_MODEL]["key"]
    matches = [(label, info) for label, info in MODELS.items() if info["key"] == wanted]
    if not matches:
        keys = ", ".join(info["key"] for info in MODELS.values())
        print(f"Unknown model '{wanted}'. Choose one of: {keys}")
        sys.exit(1)

    label, info = matches[0]
    print(f"Downloading {label}")
    print(f"Repository : {info['repo']}")
    print(f"Size       : {info['download']}")
    print(f"Saving to  : ./{MODEL_DIR}  (this can take several minutes)")

    # Creating the model downloads the files (if missing) and checks they load.
    WhisperModel(info["repo"], device="cpu", compute_type="int8", download_root=MODEL_DIR)
    print("Done. The model is cached and ready. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
