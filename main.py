#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

from audio import extract_audio, cleanup_audio
from transcriber import transcribe
from summarizer import load_model, summarize, is_model_loaded


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_video_files(input_path: str) -> list:
    """Get list of video files from input path."""
    path = Path(input_path)

    if path.is_file():
        return [path]
    elif path.is_dir():
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}
        return sorted([
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in video_extensions
        ])
    else:
        raise ValueError(f"Invalid path: {input_path}")


def get_output_name(video_path: str, input_base: str) -> str:
    """Generate output filename based on video path."""
    if os.path.isfile(input_base):
        safe_name = os.path.basename(video_path)
    else:
        rel_path = os.path.relpath(video_path, input_base)
        safe_name = rel_path.replace(os.sep, "_").replace("/", "_")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
    return os.path.splitext(safe_name)[0]


def output_exists(output_dir: str, name: str) -> bool:
    """Check if output files already exist."""
    transcript_path = os.path.join(output_dir, f"{name}_transcript.txt")
    summary_dir = os.path.join(output_dir, "summary")
    summary_path = os.path.join(summary_dir, f"{name}_summary.txt")
    return os.path.exists(transcript_path) and os.path.exists(summary_path)


def process_video(
    video_path: str,
    output_dir: str,
    whisper_model: str,
    llm_model: str,
    vad_model: str,
    no_speech_thold: float,
    verbose: bool
) -> dict:
    """Process a single video file.

    Args:
        video_path: Path to video file
        output_dir: Output directory
        whisper_model: Path to Whisper model
        llm_model: Path to LLM model
        vad_model: Optional path to VAD model
        no_speech_thold: No-speech threshold (0.0-1.0)
        verbose: Verbose mode

    Returns:
        Dict with processing results
    """
    result = {
        "video": video_path,
        "transcript": None,
        "summary": None,
        "error": None
    }

    audio_path = None

    try:
        if verbose:
            logging.info(f"Processing: {video_path}")

        audio_path = extract_audio(video_path, output_dir)
        if verbose:
            logging.info(f"Audio extracted: {audio_path}")

        transcript = transcribe(
            audio_path,
            whisper_model,
            vad_model=vad_model,
            no_speech_thold=no_speech_thold
        )
        result["transcript"] = transcript
        if verbose:
            logging.info(f"Transcription done: {len(transcript)} chars")

    except Exception as e:
        error_msg = f"Failed to extract/transcribe audio: {e}"
        logging.error(error_msg)
        result["error"] = error_msg
        return result

    finally:
        if audio_path and os.path.exists(audio_path):
            cleanup_audio(audio_path)

    if not result["transcript"]:
        result["error"] = "No speech detected in video (transcript is empty)"
        return result

    if not is_model_loaded():
        try:
            load_model(llm_model)
        except Exception as e:
            error_msg = f"Failed to load LLM: {e}"
            logging.warning(error_msg)
            result["error"] = error_msg
            return result

    try:
        summary = summarize(result["transcript"])
        result["summary"] = summary
        if verbose:
            logging.info(f"Summary done: {len(summary)} chars")
    except Exception as e:
        error_msg = f"Failed to generate summary: {e}"
        logging.warning(error_msg)
        result["error"] = error_msg

    return result


def save_result(result: dict, output_dir: str, name: str):
    """Save processing result to files."""
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    if result.get("transcript"):
        transcript_path = os.path.join(output_dir, f"{name}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["transcript"])
        logging.info(f"Saved transcript: {transcript_path}")

    if result.get("summary"):
        summary_path = os.path.join(summary_dir, f"{name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(result["summary"])
        logging.info(f"Saved summary: {summary_path}")

    if result.get("error"):
        error_path = os.path.join(output_dir, f"{name}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(result["error"])
        logging.warning(f"Saved error log: {error_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from video, transcribe with Whisper, and summarize with LLM"
    )
    parser.add_argument(
        "input",
        help="Video file or directory containing video files"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "-m", "--model",
        default="models/qwen3.5-4b-instruct-Q4_K_M.gguf",
        help="Path to LLM model (default: models/qwen3.5-4b-instruct-Q4_K_M.gguf)"
    )
    parser.add_argument(
        "--whisper-model",
        default="models/ggml-medium-q8_0.bin",
        help="Path to Whisper model (default: models/ggml-medium-q8_0.bin)"
    )
    parser.add_argument(
        "--vad-model",
        default="models/ggml-silero-v5.1.2.bin",
        help="Path to VAD model for silence filtering (default: models/ggml-silero-v5.1.2.bin)"
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering"
    )
    parser.add_argument(
        "--no-speech-thold",
        type=float,
        default=0.6,
        help="No-speech threshold 0.0-1.0 (default: 0.6, higher = more strict)"
    )
    parser.add_argument(
        "--whisper-cli",
        default="/tmp/whisper.cpp/build/bin/whisper-cli",
        help="Path to whisper-cli binary"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force reprocess even if output exists"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    vad_model = None if args.no_vad else args.vad_model
    if vad_model and not os.path.exists(vad_model):
        logging.warning(f"VAD model not found: {vad_model}, VAD disabled")
        vad_model = None

    if args.verbose:
        logging.info(f"Input: {args.input}")
        logging.info(f"Output: {args.output}")
        logging.info(f"LLM Model: {args.model}")
        logging.info(f"Whisper Model: {args.whisper_model}")
        logging.info(f"VAD Model: {vad_model or 'disabled'}")
        logging.info(f"No-speech threshold: {args.no_speech_thold}")

    os.makedirs(args.output, exist_ok=True)

    video_files = get_video_files(args.input)
    if not video_files:
        logging.error(f"No video files found in {args.input}")
        sys.exit(1)

    logging.info(f"Found {len(video_files)} video file(s)")

    from transcriber import set_whisper_cli_path
    set_whisper_cli_path(args.whisper_cli)

    skipped = 0
    succeeded = 0
    failed = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        name = get_output_name(str(video_path), os.path.abspath(args.input))

        if not args.force and output_exists(args.output, name):
            logging.info(f"Skipping (output exists): {video_path}")
            skipped += 1
            continue

        if args.verbose:
            logging.info(f"Processing: {video_path}")

        result = process_video(
            str(video_path),
            args.output,
            args.whisper_model,
            args.model,
            vad_model,
            args.no_speech_thold,
            args.verbose
        )

        save_result(result, args.output, name)

        if result.get("error"):
            logging.warning(f"Completed with error: {result['error']}")
            failed += 1
        else:
            logging.info(f"Completed successfully")
            succeeded += 1

    logging.info(
        f"All Done! Total: {len(video_files)}, "
        f"Succeeded: {succeeded}, Failed: {failed}, Skipped: {skipped}"
    )


if __name__ == "__main__":
    main()