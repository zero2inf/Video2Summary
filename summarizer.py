import os
import re
import logging
from opencc import OpenCC
from llama_cpp import Llama

logger = logging.getLogger(__name__)

llm = None
cc = OpenCC('t2s')

MAX_TRANSCRIPT_CHARS = 10000  # Traditional to Simplified Chinese

SYSTEM_PROMPT = "你是一个视频内容分析专家，擅长根据视频转写内容生成简洁、有信息量的视频简介。请务必使用简体中文输出，不要使用繁体中文。"

USER_PROMPT = """请仔细阅读以下视频转写内容，然后生成一个简洁的视频简介，包括：
1. 视频主题
2. 主要内容概述
3. 关键要点

重要：请使用简体中文（简体字）输出，不要使用繁体中文。

转写内容：
{transcript}

请直接输出视频简介。"""


def load_model(model_path: str, n_ctx: int = 16384, n_gpu_layers: int = -1):
    """Load LLM model.

    Args:
        model_path: Path to GGUF model
        n_ctx: Context length
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
    """
    global llm

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading LLM model from {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    logger.info("LLM model loaded")


def summarize(transcript: str) -> str:
    """Generate summary from transcript using LLM.

    Args:
        transcript: Transcribed text

    Returns:
        Summary text
    """
    global llm

    if llm is None:
        raise RuntimeError("LLM model not loaded. Call load_model() first.")

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty")

    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]
        logging.warning(f"Transcript truncated to {MAX_TRANSCRIPT_CHARS} characters")

    logger.info("Generating summary with LLM")

    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{USER_PROMPT.format(transcript=transcript)}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    response = llm(
        prompt,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        stop=["<|im_end|>"]
    )

    summary = response["choices"][0]["text"].strip()

    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    summary = cc.convert(summary)

    logger.info(f"Summary generated, {len(summary)} characters")
    return summary


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return llm is not None