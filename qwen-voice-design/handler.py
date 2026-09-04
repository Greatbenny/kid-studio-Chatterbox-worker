import base64
import gc
import io
import os
import time
from pathlib import Path
from typing import Any

import runpod
import soundfile as sf
import torch

SERVICE = "kid-studio-qwen-voice-design-worker"
BUILD = "qwen3-tts-voice-design-v1"
MODEL_ID = os.getenv(
    "QWEN_VOICE_DESIGN_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)
HF_HOME = Path(os.getenv("HF_HOME", "/runpod-volume/huggingface"))
TMP_ROOT = Path(os.getenv("TMPDIR", "/runpod-volume/tmp"))
MAX_TEXT_CHARS = 600
LANGUAGES = {
    "auto": "Auto", "unspecified": "Auto", "zh": "Chinese", "en": "English",
    "ja": "Japanese", "ko": "Korean", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}
_model: Any = None


def _storage() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_bytes": props.total_memory,
        "cuda": torch.version.cuda,
    }


def _load() -> Any:
    global _model
    if _model is not None:
        return _model
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    _storage()
    from qwen_tts import Qwen3TTSModel
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return _model


def _unload() -> None:
    global _model
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _voice_instruction(profile: Any) -> str:
    if not isinstance(profile, dict):
        raise ValueError("voice_profile must be an object.")
    fields = (
        ("age band", "age_band"),
        ("presentation", "presentation"),
        ("species", "species"),
        ("accent", "accent"),
        ("timbre", "timbre"),
        ("speaking style", "speaking_style"),
    )
    traits = []
    for label, key in fields:
        value = str(profile.get(key) or "").strip()
        if value and value.lower() != "unspecified":
            traits.append(f"{label}: {value}")
    if not traits:
        raise ValueError(
            "voice_profile must contain story-derived character traits."
        )
    return (
        "Create a distinctive, natural, child-safe fictional character voice. "
        + "; ".join(traits)
        + ". Keep the identity stable, intelligible, and suitable for animation."
    )


def _generate(data: dict[str, Any]) -> dict[str, Any]:
    if str(data.get("voice_creation_mode") or "").lower() != "designed":
        raise ValueError("This endpoint only accepts designed voices.")
    if data.get("reference_audio"):
        raise ValueError("Designed voice generation does not accept reference audio.")
    text = str(data.get("text") or "").strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1-{MAX_TEXT_CHARS} characters.")
    requested_language = str(data.get("language") or "Auto").strip()
    language = LANGUAGES.get(
        requested_language.lower(),
        requested_language or "Auto",
    )
    instruction = _voice_instruction(data.get("voice_profile"))
    model = _load()
    started = time.perf_counter()
    wavs, sample_rate = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruction,
    )
    output = io.BytesIO()
    sf.write(output, wavs[0], sample_rate, format="WAV")
    raw = output.getvalue()
    return {
        "ok": True,
        "service": SERVICE,
        "worker_build": BUILD,
        "model": MODEL_ID,
        "voice_creation_mode": "designed",
        "voice_cloned": False,
        "sample_rate": int(sample_rate),
        "inference_ms": round((time.perf_counter() - started) * 1000),
        "mime_type": "audio/wav",
        "audio_base64": base64.b64encode(raw).decode("ascii"),
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input")
    if not isinstance(data, dict):
        return {"ok": False, "error": "input must be an object."}
    operation = str(data.get("operation") or "generate").lower()
    try:
        if operation in {"health", "preflight"}:
            return {
                "ok": True,
                "service": SERVICE,
                "worker_build": BUILD,
                "model": MODEL_ID,
                "gpu": _gpu(),
            }
        if operation == "unload":
            _unload()
            return {"ok": True, "service": SERVICE, "unloaded": True}
        if operation == "warmup":
            _load()
            return {"ok": True, "service": SERVICE, "loaded": True}
        if operation not in {"generate", "synthesize"}:
            raise ValueError("Unsupported operation.")
        return _generate(data)
    except Exception as exc:
        return {
            "ok": False,
            "service": SERVICE,
            "worker_build": BUILD,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


if __name__ == "__main__":
    _storage()
    runpod.serverless.start({"handler": handler})
