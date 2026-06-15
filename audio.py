import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """Extract audio from video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted audio. If None, uses temp directory.

    Returns:
        Path to the extracted audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, f"{base_name}_audio.wav")
    else:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    logger.info(f"Extracting audio from {video_path}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")

    logger.info(f"Audio extracted to {audio_path}")
    return audio_path


def cleanup_audio(audio_path: str):
    """Clean up temporary audio file."""
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            logger.debug(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up audio file: {e}")