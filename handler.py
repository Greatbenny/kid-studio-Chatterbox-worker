import base64
import gc
import hashlib
import io
import os
import random
import secrets
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests
import runpod
import torch

SERVICE = "kid-studio-chatterbox-worker"
WORKER_BUILD = "chatterbox-multilingual-v3-1"
MODEL_NAME = "ResembleAI/chatterbox"
MODEL_LICENSE = "MIT"
MODEL_VARIANT = os.getenv("CHATTERBOX_T3_MODEL", "v3")
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
}
MAX_REFERENCE_BYTES = 24 * 1024 * 1024
MAX_TEXT_CHARS = 600
RELOAD_AFTER_JOBS = max(
    1,
    int(os.getenv("CHATTERBOX_RELOAD_AFTER_JOBS", "10")),
)

HF_HOME = Path(os.getenv("HF_HOME", "/runpod-volume/huggingface"))
TMP_ROOT = Path(os.getenv("TMPDIR", "/runpod-volume/tmp"))

_model: Any = None
_generation_count = 0


def _ensure_storage() -> None:
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
        "torch": torch.__version__,
    }


def _storage() -> dict[str, Any]:
    _ensure_storage()
    try:
        stat = os.statvfs("/runpod-volume")
        return {
            "root": "/runpod-volume",
            "hf_home": str(HF_HOME),
            "free_bytes": stat.f_bavail * stat.f_frsize,
            "total_bytes": stat.f_blocks * stat.f_frsize,
        }
    except OSError:
        return {
            "root": None,
            "hf_home": str(HF_HOME),
            "free_bytes": None,
            "total_bytes": None,
        }


def _health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": SERVICE,
        "worker_build": WORKER_BUILD,
        "model": MODEL_NAME,
        "model_variant": MODEL_VARIANT,
        "model_license": MODEL_LICENSE,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "voice_cloning": True,
        "voice_clone_consent_required": True,
        "watermarked": True,
        "loaded": _model is not None,
        "generations_since_load": _generation_count,
        "reload_after_jobs": RELOAD_AFTER_JOBS,
        "gpu": _gpu(),
        "storage": _storage(),
    }


def _unload() -> None:
    global _model, _generation_count
    _model = None
    _generation_count = 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    _ensure_storage()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    _model = ChatterboxMultilingualTTS.from_pretrained(
        device="cuda",
        t3_model=MODEL_VARIANT,
    )
    return _model


def _number(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if result < minimum or result > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return result


def _seed(value: Any) -> int:
    if value is None:
        return secrets.randbelow(2**31)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer.") from exc
    if result < 0 or result >= 2**31:
        raise ValueError("seed must be between 0 and 2147483647.")
    return result


def _download_reference(value: str) -> bytes:
    if value.startswith("data:audio/"):
        marker = ";base64,"
        if marker not in value:
            raise ValueError(
                "reference_audio data URL must use base64 encoding."
            )
        raw = base64.b64decode(
            value.split(marker, 1)[1],
            validate=True,
        )
    elif value.startswith("https://"):
        response = requests.get(value, timeout=45)
        response.raise_for_status()
        raw = response.content
    else:
        raw = base64.b64decode(value, validate=True)

    if not raw or len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError(
            "reference_audio must be between 1 byte and 24 MB."
        )
    return raw


def _reference_file(
    value: Any,
    consent: Any,
) -> tuple[str | None, str | None]:
    if value is None or value == "":
        return None, None
    if not isinstance(value, str):
        raise ValueError("reference_audio must be a string.")
    if consent is not True:
        raise ValueError(
            "voice_clone_consent=true is required when cloning a voice."
        )

    raw = _download_reference(value)
    digest = hashlib.sha256(raw).hexdigest()
    handle = tempfile.NamedTemporaryFile(
        dir=TMP_ROOT,
        prefix="voice-reference-",
        suffix=".audio",
        delete=False,
    )
    try:
        handle.write(raw)
        return handle.name, digest
    finally:
        handle.close()


def _wav_bytes(audio: Any, sample_rate: int) -> tuple[bytes, float]:
    if torch.is_tensor(audio):
        array = audio.detach().float().cpu().numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 1 or array.size == 0:
        raise RuntimeError("Chatterbox returned invalid audio.")
    array = np.nan_to_num(array)
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue(), array.size / float(sample_rate)


def _synthesize(data: dict[str, Any]) -> dict[str, Any]:
    global _generation_count

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"text cannot exceed {MAX_TEXT_CHARS} characters per job."
        )

    language = str(data.get("language") or "en").strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language. Supported codes: "
            + ", ".join(sorted(SUPPORTED_LANGUAGES))
        )

    seed = _seed(data.get("seed"))
    exaggeration = _number(
        data.get("exaggeration"), "exaggeration", 0.25, 2.0, 0.5
    )
    temperature = _number(
        data.get("temperature"), "temperature", 0.05, 2.0, 0.8
    )
    cfg_weight = _number(
        data.get("cfg_weight"), "cfg_weight", 0.0, 1.0, 0.5
    )
    repetition_penalty = _number(
        data.get("repetition_penalty"),
        "repetition_penalty",
        1.0,
        2.0,
        1.2,
    )

    reference_path, reference_sha = _reference_file(
        data.get("reference_audio"),
        data.get("voice_clone_consent"),
    )
    cloned = reference_path is not None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        model = _load_model()
        started = time.perf_counter()
        audio = model.generate(
            text,
            language_id=language,
            audio_prompt_path=reference_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
        )
        inference_ms = round(
            (time.perf_counter() - started) * 1000
        )
        raw, duration = _wav_bytes(audio, int(model.sr))
        _generation_count += 1
    finally:
        if reference_path:
            try:
                os.unlink(reference_path)
            except OSError:
                pass

    should_reload = _generation_count >= RELOAD_AFTER_JOBS
    result = {
        "ok": True,
        "service": SERVICE,
        "worker_build": WORKER_BUILD,
        "model": MODEL_NAME,
        "model_variant": MODEL_VARIANT,
        "model_license": MODEL_LICENSE,
        "language": language,
        "sample_rate": int(model.sr),
        "duration_seconds": round(duration, 3),
        "seed": seed,
        "voice_cloned": cloned,
        "voice_reference_sha256": reference_sha,
        "consent_asserted": cloned,
        "watermarked": True,
        "inference_ms": inference_ms,
        "mime_type": "audio/wav",
        "audio_sha256": hashlib.sha256(raw).hexdigest(),
        "audio_base64": base64.b64encode(raw).decode("ascii"),
        "model_reload_scheduled": should_reload,
    }
    if should_reload:
        _unload()
    return result


def handler(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input")
    if not isinstance(data, dict):
        return {
            "ok": False,
            "worker_build": WORKER_BUILD,
            "error": "input must be a JSON object.",
        }

    operation = str(
        data.get("operation") or "synthesize"
    ).strip().lower()

    try:
        if operation in {"health", "preflight"}:
            return _health()
        if operation == "unload":
            _unload()
            return {**_health(), "unloaded": True}
        if operation == "warmup":
            started = time.perf_counter()
            _load_model()
            return {
                **_health(),
                "load_ms": round(
                    (time.perf_counter() - started) * 1000
                ),
            }
        if operation not in {"synthesize", "generate"}:
            raise ValueError(
                "operation must be health, preflight, warmup, "
                "unload, or synthesize."
            )
        return _synthesize(data)
    except Exception as exc:
        return {
            "ok": False,
            "service": SERVICE,
            "worker_build": WORKER_BUILD,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


if __name__ == "__main__":
    _ensure_storage()
    runpod.serverless.start({"handler": handler})
