import torch
import torch.nn as nn

from comfy.audio_encoders.whisper import AudioEncoder
from comfy.text_encoders.llama import BaseGenerate, BaseQwen3, Llama2_, Qwen3_06BConfig


class VQAdaptor(nn.Module):
    def __init__(self, input_dim, hidden_size, eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.Sequential(
            operations.Linear(input_dim, hidden_size, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            operations.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.layers(x)


class MossBackbone(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        audio, text = config["audio_config"], config["text_config"]
        self.whisper_encoder = AudioEncoder(n_mels=audio["num_mel_bins"], n_ctx=audio["max_source_positions"], n_state=audio["d_model"],
                                            n_head=audio["encoder_attention_heads"], n_layer=audio["encoder_layers"], dtype=dtype, device=device, operations=operations)
        llm_config = Qwen3_06BConfig(vocab_size=text["vocab_size"], hidden_size=text["hidden_size"], intermediate_size=text["intermediate_size"],
                                     num_hidden_layers=text["num_hidden_layers"], num_attention_heads=text["num_attention_heads"],
                                     num_key_value_heads=text["num_key_value_heads"], rms_norm_eps=text["rms_norm_eps"], rope_theta=text["rope_theta"])
        llm_config.head_dim = text["head_dim"]
        self.language_model = Llama2_(llm_config, device=device, dtype=dtype, ops=operations)
        self.vq_adaptor = VQAdaptor(config["adaptor_input_dim"], text["hidden_size"], text["rms_norm_eps"], device=device, dtype=dtype, operations=operations)


class MossTranscribeDiarize(nn.Module):
    """MOSS-Transcribe-Diarize: Whisper-Medium encoder, 4x time merge + adaptor, Qwen3-0.6B decoder."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.audio_token_id = config["audio_token_id"]
        self.merge_size = config["audio_merge_size"]
        self.model = MossBackbone(config, device=device, dtype=dtype, operations=operations)

    def embed(self, input_ids, features, token_lengths):
        """features: [30 s chunks, 80, 3000] log-mel; token_lengths: audio tokens per chunk."""
        audio = []
        for i in range(0, features.shape[0], 4):
            x, _ = self.model.whisper_encoder(features[i:i + 4])
            for j, n in enumerate(token_lengths[i:i + 4]):
                audio.append(x[j, :n * self.merge_size].reshape(n, -1))
        audio = self.model.vq_adaptor(torch.cat(audio))
        embeds = self.model.language_model.embed_tokens(input_ids, out_dtype=audio.dtype)
        embeds[input_ids == self.audio_token_id] = audio
        return embeds

    def generate(self, embeds, max_new_tokens, temperature):
        lm = self.model.language_model
        out = Generator(lm).generate(embeds, do_sample=temperature > 0, max_length=max_new_tokens, temperature=temperature, top_k=0, top_p=1.0,
                                     seed=0, execution_dtype=self.dtype)
        if out and out[-1] in lm.config.stop_tokens:
            out.pop()
        return out


class Generator(BaseQwen3, BaseGenerate):
    """Comfy's KV-cached decoding loop over a Llama2_ with tied embeddings."""

    def __init__(self, model):
        self.model = model
