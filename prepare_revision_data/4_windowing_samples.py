#!/usr/bin/env python3
"""
Step 4: Window trajectory samples to limit images (keep last K screenshots).

Runs AFTER Step 3 (which produced one JSON per trajectory) and creates multiple
windowed samples per trajectory.

- Uses Step-3-style default paths under ./agentnet_data
- Reads BOTH datasets (ubuntu + win_mac) and writes ALL windowed samples into ONE
  output directory:
    ./agentnet_data/windowed_training_data/
- Output filenames are prefixed with dataset name:
    "ubuntu_{traj_id}_{sample_num}.json"
    "win_mac_{traj_id}_{sample_num}.json"

Key behavior:
- window_size=3 => at most 3 images per sample.
- For screenshots outside the window: remove "<image>" tag entirely from the
  corresponding user message (no redaction token).
- Images list order is exactly chronological, matching the kept user turns.

Input format (from Step 3):
{
  "messages": [{"role": "user"/"assistant", "content": "..."} ...],
  "images": ["path1", "path2", ...]
}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm


# -----------------------------
# IO helpers
# -----------------------------

def load_trajectory(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Windowing logic
# -----------------------------

def _collect_user_message_indices(messages: List[Dict[str, str]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "user"]


def _strip_image_tag(text: str) -> str:
    if not text:
        return text
    new = text.replace("<image>", "")
    new = new.replace("  ", " ")
    new = "\n".join(line.rstrip() for line in new.splitlines())
    return new.strip()


def create_sliding_windows(
    trajectory: Dict[str, Any],
    window_size: int = 3,
) -> List[Dict[str, Any]]:
    """
    Create one training sample per step (user+assistant pair).
    Each sample includes all conversation history up to that step, but only the
    most recent `window_size` screenshots remain with <image> tags and appear in
    images[].

    Returns list of:
      {"messages": [...], "images": [...]}
    """
    messages: List[Dict[str, str]] = trajectory.get("messages") or []
    images: List[str] = trajectory.get("images") or []

    if not messages:
        return []

    user_msg_indices = _collect_user_message_indices(messages)

    def image_for_user_turn(u_idx: int) -> str:
        return images[u_idx] if 0 <= u_idx < len(images) else ""

    num_steps = len(user_msg_indices)
    windows: List[Dict[str, Any]] = []

    for step in range(num_steps):
        # slice history up to this step's assistant (user + assistant)
        u_msg_i = user_msg_indices[step]
        end = u_msg_i + 2
        history_msgs = [m.copy() for m in messages[:end]]

        # keep only last window_size user turns
        keep_start = max(0, step + 1 - window_size)
        keep_user_turns = set(range(keep_start, step + 1))

        # images in exact chronological order of kept user turns
        window_images: List[str] = []
        for u in range(keep_start, step + 1):
            img = image_for_user_turn(u)
            if img:
                window_images.append(img)

        # remove <image> tag from older user turns
        user_seen = 0
        for m in history_msgs:
            if m.get("role") != "user":
                continue
            if user_seen not in keep_user_turns:
                m["content"] = _strip_image_tag(m.get("content", ""))
            user_seen += 1

        windows.append({"messages": history_msgs, "images": window_images})

    return windows


# -----------------------------
# Dataset processing
# -----------------------------

def process_dataset_dir(
    dataset_prefix: str,
    input_dir: Path,
    output_dir: Path,
    window_size: int,
    pattern: str = "*.json",
) -> Tuple[int, int]:
    """
    For a dataset (ubuntu or win_mac), read all trajectories in input_dir,
    write windowed samples into ONE shared output_dir, prefixing filenames
    with dataset_prefix.

    Returns: (num_trajectories_processed, total_samples_written)
    """
    input_files = sorted(input_dir.glob(pattern))
    if not input_files:
        print(f"⚠️  No input files matched {pattern} in {input_dir}")
        return 0, 0

    traj_count = 0
    sample_count = 0

    for traj_path in tqdm(input_files, desc=f"Windowing {dataset_prefix}", unit="traj"):
        traj_id = traj_path.stem
        try:
            trajectory = load_trajectory(traj_path)
            windows = create_sliding_windows(trajectory, window_size=window_size)
            if not windows:
                continue

            for sample_num, sample in enumerate(windows):
                out_path = output_dir / f"{dataset_prefix}_{traj_id}_{sample_num}.json"
                save_json(sample, out_path)
                sample_count += 1

            traj_count += 1
        except Exception as e:
            print(f"\n⚠️  Error processing {traj_path}: {e}")

    return traj_count, sample_count


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Window AgentNet Step-3 trajectories into per-step samples (keep last K images)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./agentnet_data",
        help="Base data directory (default: ./agentnet_data).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="How many most-recent screenshots to keep (default: 3).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for Step-3 trajectory files (default: *.json).",
    )
    parser.add_argument(
        "--ubuntu-in",
        type=str,
        default=None,
        help="Override Ubuntu Step-3 dir (default: {data-dir}/ubuntu_training_data).",
    )
    parser.add_argument(
        "--win-mac-in",
        type=str,
        default=None,
        help="Override Win/Mac Step-3 dir (default: {data-dir}/win_mac_training_data).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output dir (default: {data-dir}/windowed_training_data).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ubuntu_in = Path(args.ubuntu_in) if args.ubuntu_in else data_dir / "ubuntu_training_data"
    win_mac_in = Path(args.win_mac_in) if args.win_mac_in else data_dir / "win_mac_training_data"
    out_dir = Path(args.out) if args.out else data_dir / "windowed_training_data"

    if not data_dir.exists():
        raise SystemExit(f"❌ Data directory not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Step 4: Windowing samples (ONE output folder)")
    print("=" * 60)
    print(f"Base data dir: {data_dir}")
    print(f"Ubuntu input:  {ubuntu_in}")
    print(f"Win/Mac input: {win_mac_in}")
    print(f"Output dir:    {out_dir}")
    print(f"Window size:   {args.window_size}")
    print("=" * 60 + "\n")

    total_traj = 0
    total_samples = 0

    if ubuntu_in.exists():
        t, s = process_dataset_dir(
            dataset_prefix="ubuntu",
            input_dir=ubuntu_in,
            output_dir=out_dir,
            window_size=args.window_size,
            pattern=args.pattern,
        )
        total_traj += t
        total_samples += s
        print(f"\n✅ Ubuntu: processed {t} trajectories -> wrote {s} samples\n")
    else:
        print(f"⚠️  Ubuntu input dir not found, skipping: {ubuntu_in}\n")

    if win_mac_in.exists():
        t, s = process_dataset_dir(
            dataset_prefix="win_mac",
            input_dir=win_mac_in,
            output_dir=out_dir,
            window_size=args.window_size,
            pattern=args.pattern,
        )
        total_traj += t
        total_samples += s
        print(f"\n✅ Win/Mac: processed {t} trajectories -> wrote {s} samples\n")
    else:
        print(f"⚠️  Win/Mac input dir not found, skipping: {win_mac_in}\n")

    print("=" * 60)
    print("✅ Step 4 complete")
    print("=" * 60)
    print(f"Total trajectories processed: {total_traj}")
    print(f"Total windowed samples written: {total_samples}")
    print(f"All outputs in: {out_dir}")
    print()


if __name__ == "__main__":
    main()
