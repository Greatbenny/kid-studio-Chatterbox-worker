# Kid Studio Qwen3-TTS VoiceDesign worker

RunPod Serverless worker for creating new fictional character voices from
structured, story-derived voice profiles. It does not accept reference audio
and does not clone a person.

Recommended GPU: 24 GB or larger. Attach a network volume so Hugging Face model
weights persist between cold starts.

The application sends `voice_creation_mode=designed`, audition `text`,
`language`, and a structured `voice_profile`.
