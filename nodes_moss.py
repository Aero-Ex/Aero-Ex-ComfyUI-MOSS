import json

import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import torch
from tokenizers import Tokenizer

from .moss.features import extract_features, resample_16k
from .moss.model import MossTranscribeDiarize
from .moss.prompt import DEFAULT_PROMPT, prompt_ids
from .moss.srt import segments_to_json, segments_to_raw, segments_to_srt
from .moss.transcript import parse_transcript


class MOSSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "moss_name": (folder_paths.get_filename_list("mossd"), {
                "tooltip": "Single-file MOSS-Transcribe-Diarize checkpoint."}),
        }}

    RETURN_TYPES = ("MOSSDIARIZE",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "MOSSDiarize"

    def execute(self, moss_name):
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("mossd", moss_name), safe_load=True, return_metadata=True)
        metadata = metadata or {}
        if (metadata.get("mossd.format_version") != "1"
                or "model.language_model.embed_tokens.weight" not in sd
                or "model.whisper_encoder.conv1.weight" not in sd
                or "model.vq_adaptor.layers.0.weight" not in sd):
            raise RuntimeError("Not a single-file MOSS-Transcribe-Diarize checkpoint. Convert one with tools/convert_moss_checkpoint.py.")
        config = json.loads(metadata["mossd.config"])

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        # Same dtype policy as core's model loading: bf16 where supported, fp16 on older GPUs, honoring --force-fp16 and similar flags.
        supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        # INT8 checkpoints carry per-layer .comfy_quant keys, consumed by the mixed-precision ops.
        quant = comfy.utils.detect_layer_quantization(sd, "")
        weight_dtype = None if quant is not None else comfy.utils.weight_dtype(sd)
        dtype = comfy.model_management.unet_dtype(model_params=comfy.utils.calculate_parameters(sd), supported_dtypes=supported_dtypes, weight_dtype=weight_dtype)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(None if quant is not None else dtype, load_device, supported_dtypes)
        if quant is not None:
            operations = comfy.ops.mixed_precision_ops(quant, manual_cast_dtype or dtype)
        else:
            operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device)
            if weight_dtype != dtype:
                # Lazily loaded layers keep the checkpoint tensors, so convert them to the model dtype once here.
                sd = {k: v.to(dtype) if v.dtype == weight_dtype else v for k, v in sd.items()}

        model = MossTranscribeDiarize(config, device=offload_device, dtype=dtype, operations=operations)
        model.load_state_dict(sd)
        model.dtype = manual_cast_dtype or dtype
        # Plain ModelPatcher: decoding ran ~1.5x slower under dynamic VRAM.
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        model.tokenizer = Tokenizer.from_str(metadata["mossd.tokenizer"])
        return (patcher,)


class MOSSTranscribeDiarize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MOSSDIARIZE", {"tooltip": "Loaded MOSS-Transcribe-Diarize model."}),
            "audio": ("AUDIO", {"tooltip": "Audio to transcribe (resampled to 16kHz mono)."}),
            "max_new_tokens": ("INT", {"default": 5120, "min": 256, "max": 65536, "step": 256, "tooltip": "Maximum tokens generated per audio."}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "0 is greedy decoding, as upstream."}),
        }, "optional": {
            "system_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Empty uses the upstream diarization prompt."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "segments_json", "srt")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_TOOLTIPS = ("One entry per audio in the batch.",) * 3
    FUNCTION = "execute"
    CATEGORY = "MOSSDiarize"

    def execute(self, model, audio, max_new_tokens=5120, temperature=0.0, system_prompt=""):
        comfy.model_management.load_models_gpu([model])
        moss = model.model
        prompt = system_prompt.strip() or DEFAULT_PROMPT
        out = ([], [], [])
        for wav in resample_16k(audio["waveform"], audio["sample_rate"]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            raw = ""
            if wav.abs().max() >= 1e-3:  # the model invents text for silent audio
                features, token_lengths = extract_features(wav)
                ids = torch.tensor([prompt_ids(moss.tokenizer, prompt, sum(token_lengths), moss.audio_token_id)], device=model.load_device)
                embeds = moss.embed(ids, features.to(model.load_device, moss.dtype), token_lengths)
                raw = moss.tokenizer.decode(moss.generate(embeds, max_new_tokens, temperature), skip_special_tokens=True).strip()
            segments = parse_transcript(raw)
            out[0].append(segments_to_raw(segments) if segments else raw)
            out[1].append(segments_to_json(segments))
            out[2].append(segments_to_srt(segments))
        return out


NODE_CLASS_MAPPINGS = {
    "MOSSLoader": MOSSLoader,
    "MOSSTranscribeDiarize": MOSSTranscribeDiarize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoader": "Load MOSS-Diarize Model",
    "MOSSTranscribeDiarize": "MOSS Transcribe + Diarize",
}
