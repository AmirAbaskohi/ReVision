"""
Step 3: Generate training JSON files from AgentNet data
Processes ALL samples from both Ubuntu and Windows/Mac datasets.

One output JSON per trajectory:
{
  "messages": [... user/assistant alternating ...],
  "images": ["...png", "...png", ...]   # EXACT same order as steps in traj
}

Rules:
- NO raw observation text, NO metadata, NO raw pyautogui code string in the output.
- For each step with an image and a parseable action:
    user:      "## Task: {instruction}\n\nWe are now on this page. What should we do next?\n<image>"
    assistant: "{reasoning}\n<tool_call>\n{tool call json}\n</tool_call>"
      where reasoning = reflection (reflects on the prior step) followed by thought,
      as plain prose with no "## Thought:"/"## Action:" headers, and the tool call is
      {"name": "computer_use", "arguments": {...}} with pixel-space "coordinate" fields
      (converted from AgentNet's normalized [0,1] x/y using the actual screenshot's
      resolution).
- Steps whose "code" can't be parsed into a supported computer_use action are dropped
  entirely (no user/assistant messages, no image added for that step) rather than
  failing the whole trajectory.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from PIL import Image
except ImportError:  # Pillow is required to read screenshot resolution for coordinates
    Image = None


USER_PROMPT_TEMPLATE = (
    "## Task: {instruction}\n\n"
    "We are now on this page. What should we do next?\n"
    "<image>"
)


# ---------------------------------------------------------------------------
# Action parsing: AgentNet pyautogui/computer "code" string -> "computer_use"
# tool call arguments.
# ---------------------------------------------------------------------------

def _full_call_name(call: ast.Call) -> Optional[str]:
    """Return e.g. 'pyautogui.click' / 'computer.terminate' / 'press' for a Call node."""
    func = call.func
    if isinstance(func, ast.Attribute):
        parts = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(func, ast.Name):
        return func.id
    return None


def _call_args(call: ast.Call) -> Tuple[List[Any], Dict[str, Any]]:
    """Extract positional/keyword args as plain Python literals.

    Raises ValueError if any argument isn't a literal (e.g. a variable reference),
    which the caller treats as "unsupported -> drop this step".
    """
    args = [ast.literal_eval(a) for a in call.args]
    kwargs: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("unsupported **kwargs")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return args, kwargs


def _extract_calls(code: str) -> Optional[List[ast.Call]]:
    """Parse a code string into its top-level Call nodes (one per statement)."""
    try:
        tree = ast.parse(code.strip(), mode="exec")
    except SyntaxError:
        return None

    calls: List[ast.Call] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            calls.append(stmt.value)
        else:
            return None  # non-call statement -> unsupported
    return calls if calls else None


def _to_pixels(x: float, y: float, width: int, height: int) -> List[int]:
    """Convert normalized [0,1] coordinates to pixel coordinates using image size."""
    if 0 <= x <= 1 and 0 <= y <= 1:
        return [round(x * width), round(y * height)]
    return [round(x), round(y)]


def _get_xy(args: List[Any], kwargs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    x = kwargs.get("x")
    y = kwargs.get("y")
    if x is None and len(args) >= 1:
        x = args[0]
    if y is None and len(args) >= 2:
        y = args[1]
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _click_type(func_name: str, kwargs: Dict[str, Any]) -> str:
    button = str(kwargs.get("button", "left")).lower()
    clicks = kwargs.get("clicks", 1)
    if func_name.endswith("doubleClick") or clicks == 2:
        return "double_click"
    if func_name.endswith("rightClick") or button == "right":
        return "right_click"
    if func_name.endswith("middleClick") or button == "middle":
        return "middle_click"
    return "left_click"


def parse_action_code(
    code: str,
    img_width: Optional[int],
    img_height: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Convert an AgentNet 'code' string into the "arguments" payload of a
    {"name": "computer_use", "arguments": {...}} tool call.

    Returns None if the code can't be parsed or maps to no supported action; the
    caller must drop that trajectory step in this case.
    """
    if not code or not code.strip():
        return None

    calls = _extract_calls(code)
    if not calls:
        return None

    # Special case: write(...) immediately chained with press('enter') in the SAME
    # code string -> a single "type" action with press_enter=True.
    if len(calls) == 2:
        first_name = _full_call_name(calls[0]) or ""
        second_name = _full_call_name(calls[1]) or ""
        if first_name.endswith(("write", "typewrite")) and second_name.endswith("press"):
            try:
                f_args, f_kwargs = _call_args(calls[0])
                s_args, s_kwargs = _call_args(calls[1])
            except ValueError:
                return None
            text = f_kwargs.get("text", f_args[0] if f_args else None)
            key = s_kwargs.get("keys", s_args[0] if s_args else None)
            if isinstance(text, str) and isinstance(key, str) and key.lower() == "enter":
                return {"type": "type", "text": text, "press_enter": True, "delete_existing_text": False}
        return None  # any other multi-statement combo is unsupported

    if len(calls) != 1:
        return None

    call = calls[0]
    func_name = _full_call_name(call)
    if not func_name:
        return None

    try:
        args, kwargs = _call_args(call)
    except ValueError:
        return None

    short_name = func_name.split(".")[-1]

    # --- Mouse actions requiring a coordinate ---
    if short_name in ("click", "doubleClick", "rightClick", "middleClick", "moveTo", "dragTo"):
        xy = _get_xy(args, kwargs)
        if xy is None or img_width is None or img_height is None:
            return None
        coordinate = _to_pixels(xy[0], xy[1], img_width, img_height)

        if short_name == "moveTo":
            return {"type": "mouse_move", "coordinate": coordinate}
        if short_name == "dragTo":
            return {"type": "left_click_drag", "coordinate": coordinate}

        return {"type": _click_type(func_name, kwargs), "coordinate": coordinate}

    # --- Typing ---
    if short_name in ("write", "typewrite"):
        text = kwargs.get("text", args[0] if args else None)
        if not isinstance(text, str):
            return None
        return {"type": "type", "text": text, "press_enter": False, "delete_existing_text": False}

    # --- Key press / hotkey ---
    if short_name == "press":
        key = kwargs.get("keys", args[0] if args else None)
        if isinstance(key, (list, tuple)):
            keys = [str(k) for k in key]
        elif isinstance(key, str):
            keys = [key]
        else:
            return None
        return {"type": "key", "keys": keys}

    if short_name == "hotkey":
        # pyautogui.hotkey('ctrl', 'o')  OR  pyautogui.hotkey(['ctrl', 'o'])
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            keys = [str(k) for k in args[0]]
        else:
            keys = [str(a) for a in args]
        if not keys:
            return None
        return {"type": "key", "keys": keys}

    # --- Scroll ---
    if short_name == "scroll":
        amount = kwargs.get("clicks", args[0] if args else None)
        if amount is None:
            return None
        try:
            result: Dict[str, Any] = {"type": "scroll", "amount": int(amount)}
        except (TypeError, ValueError):
            return None
        x, y = kwargs.get("x"), kwargs.get("y")
        if x is not None and y is not None and img_width is not None and img_height is not None:
            result["coordinate"] = _to_pixels(float(x), float(y), img_width, img_height)
        return result

    # --- Terminate ---
    if func_name in ("computer.terminate", "terminate"):
        status = kwargs.get("status", args[0] if args else None)
        if not isinstance(status, str):
            return None
        return {"type": "terminate", "status": status}

    # Unsupported / unmapped function
    return None


def convert_trajectory_to_llamafactory_format(
    task_data: Dict[str, Any],
    image_base_dir: Path,
    *,
    require_image: bool = True,
    filter_incorrect: bool = False,
    filter_redundant: bool = False,
    skip_missing_images: bool = True,
) -> Dict[str, Any]:
    """
    Convert a single AgentNet trajectory to LLaMA-Factory multi-image conversation format.

    - One user+assistant pair per trajectory step (only for steps that have an image
      AND whose "code" parses into a supported computer_use tool call).
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

        code = (v.get("code") or "").strip()
        if not code:
            continue

        img_path = image_base_dir / img_name
        if skip_missing_images and not img_path.exists():
            continue

        img_width: Optional[int] = None
        img_height: Optional[int] = None
        if Image is not None:
            try:
                with Image.open(img_path) as im:
                    img_width, img_height = im.size
            except Exception:
                img_width = img_height = None

        arguments = parse_action_code(code, img_width, img_height)
        if arguments is None:
            # Unsupported / unparseable action -> drop this step entirely.
            continue

        tool_call = {"name": "computer_use", "arguments": arguments}

        # Reasoning: reflection (reflects on the prior step) followed by thought,
        # as plain prose with no headers.
        reflection = (v.get("reflection") or "").strip()
        thought = (v.get("thought") or "").strip()
        reasoning = " ".join(p for p in (reflection, thought) if p)

        # Build user prompt for THIS step (must include <image>)
        messages.append({"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction)})
        images.append(str(img_path))

        assistant_parts: List[str] = []
        if reasoning:
            assistant_parts.append(reasoning)
        assistant_parts.append(
            f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"
        )
        messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

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

    if Image is None:
        print("\n❌ Error: Pillow is required to read screenshot resolution for coordinates.")
        print("   Install with: pip3 install Pillow")
        sys.exit(1)

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
