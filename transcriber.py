import os
import subprocess
import logging

logger = logging.getLogger(__name__)


WHISPER_CLI_PATH = "/tmp/whisper.cpp/build/bin/whisper-cli"


def transcribe(
    audio_path: str,
    model_path: str,
    language: str = "zh",
    threads: int = 4,
    vad_model: str = None,
    no_speech_thold: float = 0.6
) -> str:
    """Transcribe audio file using Whisper GGML model.

    Args:
        audio_path: Path to the audio file
        model_path: Path to the Whisper GGML model
        language: Language code (default: zh for Chinese)
        threads: Number of threads to use
        vad_model: Optional path to VAD model for filtering silence
        no_speech_thold: Threshold for no-speech detection (0.0-1.0)

    Returns:
        Transcribed text
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(WHISPER_CLI_PATH):
        raise RuntimeError(
            f"Whisper CLI not found at {WHISPER_CLI_PATH}. "
            "Please compile whisper.cpp first."
        )

    cmd = [
        WHISPER_CLI_PATH,
        "-m", model_path,
        "-f", audio_path,
        "-l", language,
        "-otxt",
        "-of", audio_path,
        "--no-timestamps",
        "-nth", str(no_speech_thold),
        "-t", str(threads)
    ]

    if vad_model and os.path.exists(vad_model):
        cmd.extend(["--vad", "--vad-model", vad_model])
        logger.info(f"VAD enabled with model: {vad_model}")

    logger.info(f"Transcribing {audio_path} with Whisper")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Transcription failed: {result.stderr}")

    txt_path = audio_path + ".txt"

    if not os.path.exists(txt_path):
        raise RuntimeError(f"Transcription output not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    os.remove(txt_path)

    logger.info(f"Transcription completed, {len(text)} characters")
    return text


def set_whisper_cli_path(path: str):
    """Set custom Whisper CLI path."""
    global WHISPER_CLI_PATH
    WHISPER_CLI_PATH = path