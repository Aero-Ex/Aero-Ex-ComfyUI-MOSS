import torch

import comfy.audio

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 480000  # one 30 s Whisper window
TOKEN_SAMPLES = 160 * 2 * 4  # hop length * Whisper encoder stride * audio merge size


def resample_16k(waveform, sample_rate):
    """[batch, channels, samples] -> list of mono 16 kHz float32 waveforms on CPU."""
    waveform = waveform.mean(dim=1).float().cpu()
    # Close to soxr_hq, which upstream resamples with; the default filter noticeably changes transcripts.
    return list(comfy.audio.resample(waveform, sample_rate, SAMPLE_RATE, lowpass_filter_width=64, rolloff=0.955))


def extract_features(waveform):
    """Mono 16 kHz waveform -> ([30 s chunks, 80, 3000] Whisper log-mel, audio tokens per chunk)."""
    chunks = waveform.split(CHUNK_SAMPLES)
    token_lengths = [(c.shape[0] - 1) // TOKEN_SAMPLES + 1 for c in chunks]
    batch = torch.stack([torch.nn.functional.pad(c, (0, CHUNK_SAMPLES - c.shape[0])) for c in chunks])
    mel = comfy.audio.MelSpectrogram(SAMPLE_RATE, 400, 160, 80, f_max=8000.0)(batch)[..., :-1]
    log_spec = mel.clamp(min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    return (log_spec + 4.0) / 4.0, token_lengths
