"""Convert HF MOSS-Transcribe-Diarize folder to a single-file ComfyUI checkpoint.

Usage:
    python tools/convert_moss_checkpoint.py --hf-dir /path/to/hf --out /path/to/out.safetensors
"""

import argparse
import json
import os

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.hf_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(args.hf_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    index_path = os.path.join(args.hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    sd = {}
    for shard in shards:
        sd.update(load_file(os.path.join(args.hf_dir, shard), device="cpu"))

    metadata = {
        "mossd.format_version": "1",
        "mossd.config": json.dumps(config, ensure_ascii=False),
        "mossd.tokenizer": json.dumps(tokenizer, ensure_ascii=False),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(sd, args.out, metadata=metadata)
    print("Wrote {} ({} tensors).".format(args.out, len(sd)))


if __name__ == "__main__":
    main()
