# ComfyUI-MOSS-Diarize

dependency-free MOSS-Transcribe-Diarize speech-to-text with speaker labels for ComfyUI.

## Model

Download [`moss-transcribe-diarize-0.9B.safetensors`](https://huggingface.co/Aero-Ex/ComfyUI-MOSS-Diarize/resolve/main/moss-transcribe-diarize-0.9B.safetensors)
(1.8 GB) from [Aero-Ex/ComfyUI-MOSS-Diarize](https://huggingface.co/Aero-Ex/ComfyUI-MOSS-Diarize) into `ComfyUI/models/mossd/`.

## Usage

`Load MOSS-Diarize Model` -> `MOSS Transcribe + Diarize`. Leave `system_prompt` empty for the upstream diarization prompt.

Outputs: `transcription` (`[start][Sxx]text[end]` stream), `segments_json`, `srt` (`Sxx: text` cues).
# Aero-Ex-ComfyUI-MOSS
