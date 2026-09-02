# Kid Studio Chatterbox RunPod Worker

RunPod Serverless multilingual speech and consent-gated voice-cloning worker for Kid Studio.

## Model

- Resemble AI Chatterbox Multilingual V3
- 23 supported languages
- MIT-licensed code/model family
- Zero-shot voice cloning from reference audio
- Chatterbox implicit output watermark retained

## RunPod deployment

- Branch: `main`
- Dockerfile path: `/Dockerfile`
- Endpoint type: Queue
- GPU: 16 GB minimum; 24 GB recommended
- Minimum workers: 0
- Maximum workers: 1 initially
- Network volume mount: `/runpod-volume`

## Operations

- `health` or `preflight`: diagnostics without loading the model
- `warmup`: download/cache and load Multilingual V3
- `synthesize`: generate WAV speech
- `unload`: release model VRAM

A request with `reference_audio` must also include `voice_clone_consent: true`. The reference can be raw base64, an audio data URL, or an HTTPS URL. Kid Studio must collect and retain the speaker or guardian authorization before asserting this flag.

The response contains WAV audio, hashes, duration, model version, cloning/consent status, watermark status and inference time. RunPod adds total execution time for actual GPU-cost accounting.
