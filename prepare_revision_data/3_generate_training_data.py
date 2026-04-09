"""
Step 3: Generate training JSON files from AgentNet data
Processes ALL samples from both Ubuntu and Windows/Mac datasets.

One output JSON per trajectory:
{
  "messages": [... user/assistant alternating ...],
  "images": ["...png", "...png", ...]   # EXACT same order as steps in traj
}

Rules:
- NO observation
- NO metadata
- NO code
- For each step with an image:
    user:  Task + "We are now on this page. What should we do next?\n<image>"
    assistant: "## Thought: ...\n\n## Action: ..."
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import sys


USER_PROMPT_TEMPLATE = (
    "## Task: {instruction}\n\n"
    "We are now on this page. What should we do next?\n"
    "<image>"
)


def convert_trajectory_to_llamafactory_format(
    task_data: Dict[str, Any],
    image_base_dir: Path,
    *,
    require_image: bool = True,
    require_action: bool = True,
    filter_incorrect: bool = False,
    filter_redundant: bool = False,
    skip_missing_images: bool = True,
) -> Dict[str, Any]:
    """
    Convert a single AgentNet trajectory to LLaMA-Factory multi-image conversation format.

    - One user+assistant pair per trajectory step (only for steps that have an image).
    - images[] is appended in EXACT same order as the trajectory steps processed.
    """
    instruction = (task_data.get("instruction") or task_data.get("natural_language_task") or "").strip()
    if not instruction:
        return {"messages": [], "images": []}

    traj = task_data.get("traj", []) or []

    messages: List[Dict[str, str]] = []
    images: List[str] = []

    for step in traj:
        v = step.get("value") or {}
        img_name = (step.get("image") or "").strip()

        # Optional filtering
        if filter_incorrect and (v.get("last_step_correct") is False):
            continue
        if filter_redundant and (v.get("last_step_redundant") is True):
            continue

        if require_image and not img_name:
            continue

        action = (v.get("code") or "").strip()
        if require_action and not action:
            continue

        # Resolve image path and optionally skip missing files
        if img_name:
            img_path = image_base_dir / img_name
            if skip_missing_images and not img_path.exists():
                continue
            img_path_str = str(img_path)
        else:
            img_path_str = ""

        # Build user prompt for THIS step (must include <image>)
        user_content = USER_PROMPT_TEMPLATE.format(instruction=instruction)
        messages.append({"role": "user", "content": user_content})

        # Append image in the SAME loop iteration -> guarantees order
        if img_path_str:
            images.append(img_path_str)

        # Build assistant content: reasoning + action (no code, no observation)
        reasoning = (v.get("thought") or v.get("reflection") or "").strip()
        assistant_parts: List[str] = []
        if reasoning:
            assistant_parts.append(f"## Thought: {reasoning}")
        else:
            assistant_parts.append("## Thought:")
        assistant_parts.append(f"## Action: {action}")

        messages.append({"role": "assistant", "content": "\n\n".join(assistant_parts).strip()})

    return {"messages": messages, "images": images}


def process_dataset(dataset_name: str, data_dir: Path):
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}\n")

    if dataset_name == "ubuntu":
        jsonl_path = data_dir / "agentnet_ubuntu_5k.jsonl"
    else:
        jsonl_path = data_dir / "agentnet_win_mac_18k.jsonl"

    image_base_dir = data_dir / f"{dataset_name}_images_extracted"
    output_dir = data_dir / f"{dataset_name}_training_data"

    if not jsonl_path.exists():
        print(f"❌ Error: {jsonl_path} not found!")
        print("   Run ./1_download_agentnet.sh first!")
        return

    if not image_base_dir.exists():
        print(f"❌ Error: {image_base_dir} not found!")
        print("   Run ./2_extract_images.sh first!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Input: {jsonl_path}")
    print(f"🖼️  Images: {image_base_dir}")
    print(f"💾 Output: {output_dir}\n")

    print("📊 Counting samples...")
    with open(jsonl_path, "r") as f:
        total_lines = sum(1 for _ in f)

    print(f"📝 Processing {total_lines} trajectories...\n")

    processed = 0
    skipped = 0

    with open(jsonl_path, "r") as f:
        with tqdm(total=total_lines, desc=f"Converting {dataset_name}", unit="traj") as pbar:
            for idx, line in enumerate(f):
                try:
                    task_data = json.loads(line)
                    task_id = (task_data.get("task_id") or f"task_{idx}").strip()

                    out = convert_trajectory_to_llamafactory_format(
                        task_data,
                        image_base_dir,
                        require_image=True,
                        require_action=True,
                        filter_incorrect=False,   # set True if you want to drop incorrect steps
                        filter_redundant=False,   # set True if you want to drop redundant steps
                        skip_missing_images=True, # skips steps whose image file is missing
                    )

                    # Skip if no messages / no images produced
                    if not out["messages"] or not out["images"]:
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Sanity: images count should equal number of user turns (each user has <image>)
                    # because we add one image per user message.
                    # If you want, you can enforce this strictly:
                    # num_user = sum(1 for m in out["messages"] if m["role"] == "user")
                    # if num_user != len(out["images"]): raise ValueError("user/images mismatch")

                    output_file = output_dir / f"{task_id}.json"
                    with open(output_file, "w") as out_f:
                        json.dump(out, out_f, indent=2, ensure_ascii=False)

                    processed += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"\n⚠️  Error processing trajectory {idx}: {e}")
                    skipped += 1
                    pbar.update(1)

    print(f"\n✅ {dataset_name.upper()} processing complete!")
    print(f"   Processed: {processed} trajectories")
    print(f"   Skipped:   {skipped} trajectories")
    print(f"   Output: {output_dir}\n")


def main():
    print("\n" + "=" * 60)
    print("  Step 3: Generate LLaMA-Factory Training Data (AgentNet)")
    print("=" * 60)

    data_dir = Path("./agentnet_data")
    if not data_dir.exists():
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print("   Run ./1_download_agentnet.sh first!")
        sys.exit(1)

    process_dataset("ubuntu", data_dir)
    process_dataset("win_mac", data_dir)

    print("=" * 60)
    print("  ✅ All Processing Complete!")
    print("=" * 60)
    print("\n🎉 Your training data is ready!")
    print("\n📁 Training data locations:")
    print(f"   - {data_dir}/ubuntu_training_data/")
    print(f"   - {data_dir}/win_mac_training_data/")
    print()


if __name__ == "__main__":
    main()
