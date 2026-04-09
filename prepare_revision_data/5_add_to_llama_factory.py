#!/usr/bin/env python3
"""
5_add_to_llama_factory.py

Reads all windowed training sample JSON files from:
  agentnet_data/windowed_training_data/

Aggregates them into sharded JSONL files (LLaMA-Factory multi-modal format) and
rewrites image paths by prefixing "../../" so that:

  agentnet_data/.../file.png
-> ../../agentnet_data/.../file.png

Creates sharded JSONL files with at most 10k samples per file:
  agentnet_windowed.shard00001.jsonl
  agentnet_windowed.shard00002.jsonl
  ...

Saves output to:
  ../llama-factory-with-removal/data/agentnet_windowed_shards/
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm


def prefix_image_path(p: str, prefix: str = "../../") -> str:
    """Prefix image paths unless they are already prefixed."""
    # Normalize slashes (optional, but helps consistency)
    p = p.replace("\\", "/")
    if p.startswith(prefix):
        return p
    return prefix + p


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    input_dir = Path("agentnet_data/windowed_training_data")
    output_dir = Path("../llama-factory-with-removal/data/agentnet_windowed_shards")
    shard_size = 10000

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect JSON files (stable order)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    aggregated: List[Dict[str, Any]] = []
    skipped = 0

    for jf in tqdm(json_files, desc="Aggregating windowed samples", unit="file"):
        try:
            sample = load_json(jf)

            # Expecting: {"messages": [...], "images": [...]}
            messages = sample.get("messages")
            images = sample.get("images", [])

            if not isinstance(messages, list) or not messages:
                skipped += 1
                continue

            if not isinstance(images, list):
                images = []

            # Rewrite image paths
            new_images = [prefix_image_path(str(img)) for img in images]

            aggregated.append({
                "messages": messages,
                "images": new_images
            })

        except Exception as e:
            print(f"\n⚠️  Skipping {jf.name} due to error: {e}")
            skipped += 1

    # Write sharded JSONL files
    num_shards = (len(aggregated) + shard_size - 1) // shard_size

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(aggregated))
        shard_samples = aggregated[start_idx:end_idx]

        # Format shard filename with zero-padding
        shard_filename = f"agentnet_windowed.shard{shard_idx:05d}.jsonl"
        shard_path = output_dir / shard_filename

        with shard_path.open("w", encoding="utf-8") as f:
            for sample in shard_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("✅ Sharding complete!")
    print(f"Input dir:     {input_dir}")
    print(f"Files read:    {len(json_files)}")
    print(f"Total samples: {len(aggregated)}")
    print(f"Skipped:       {skipped}")
    print(f"Shard size:    {shard_size} samples per file")
    print(f"Num shards:    {num_shards}")
    print(f"Output dir:    {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
