"""
Utility functions for working with GUIAct dataset.
Provides tools to download, group, and visualize temporal sequences.
"""

import base64
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt


def download_guiact_files(split: str = "train", cache_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Download GUIAct JSON and Parquet files from HuggingFace.
    
    Args:
        split: Dataset split to download ("train" or "test")
        cache_dir: Optional cache directory for downloads
        
    Returns:
        Tuple of (json_path, parquet_path)
    """
    repo_id = "yiye2023/GUIAct"
    
    json_filename = f"web-multi_{split}_data.json"
    parquet_filename = f"web-multi_{split}_images.parquet"
    
    print(f"Downloading {json_filename} from {repo_id}...")
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=json_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    
    print(f"Downloading {parquet_filename} from {repo_id}...")
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    
    print(f"✓ Downloaded JSON: {json_path}")
    print(f"✓ Downloaded Parquet: {parquet_path}")
    
    return json_path, parquet_path


def parse_uid(uid: str) -> Tuple[str, str]:
    """
    Parse uid to extract image_id and qa_number.
    
    Format: uid_image_{image_id}_qa_{qa_num}
    Example: uid_image_dffe7794-aa20-48fa-98e9-c4342caa93de_qa_01
    
    Args:
        uid: UID string from dataset
        
    Returns:
        Tuple of (image_id, qa_number)
    """
    match = re.match(r'uid_image_([a-f0-9\-]+)_qa_(\d+)', uid)
    if match:
        image_id = match.group(1)
        qa_num = match.group(2)
        return image_id, qa_num
    raise ValueError(f"Invalid uid format: {uid}")


def group_by_image_id(json_path: str) -> Tuple[List[List[str]], dict]:
    """
    Group UIDs by image_id to create temporal sequences.
    
    Args:
        json_path: Path to the JSON data file
        
    Returns:
        Tuple of (grouped_step_images, statistics)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    groups = defaultdict(list)  # image_id -> list of (qa_num, uid)
    bad = 0
    
    for ex in data:
        uid = ex.get("uid", "")
        try:
            image_id, qa_num = parse_uid(uid)
            groups[image_id].append((qa_num, uid))
        except ValueError:
            bad += 1
            continue
    
    # Sort by qa_num within each image_id group
    grouped_step_images = []
    for image_id, items in groups.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        grouped_step_images.append([uid for _, uid in items_sorted])
    
    num_uids_total = len(data)
    num_groups = len(grouped_step_images)
    steps_per_group = [len(g) for g in grouped_step_images]
    avg_steps = sum(steps_per_group) / num_groups if num_groups else 0
    
    stats = {
        'total_entries': num_uids_total,
        'num_groups': num_groups,
        'avg_steps': avg_steps,
        'unmatched': bad,
        'steps_per_group': steps_per_group,
    }
    
    return grouped_step_images, stats


def uid_to_image_id(uid: str) -> str:
    """
    Extract image_id from uid.
    
    Args:
        uid: UID string
        
    Returns:
        image_id
    """
    image_id, _ = parse_uid(uid)
    return image_id


def decode_image_from_parquet(df_images: pd.DataFrame, image_id: str) -> Image.Image:
    """
    Decode image from parquet dataframe using image_id.
    
    Args:
        df_images: Parquet dataframe with base64 encoded images
        image_id: Image ID to retrieve
        
    Returns:
        PIL Image
    """
    if image_id in df_images.index:
        b64 = df_images.loc[image_id, "base64"]
    else:
        # Fallback: search in dataframe
        row = df_images[df_images.index == image_id].iloc[0]
        b64 = row["base64"]
    
    # base64 string -> bytes -> PIL
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def visualize_group(
    group_uids: List[str],
    df_images: pd.DataFrame,
    group_idx: int = 0,
    cols: int = 4,
) -> None:
    """
    Visualize a group of images in a grid.
    
    Args:
        group_uids: List of UIDs in the group
        df_images: Parquet dataframe with images
        group_idx: Index of group being visualized
        cols: Number of columns in grid
    """
    print(f"Showing group {group_idx} with {len(group_uids)} QA pairs")
    print("First few uids:", group_uids[:5])
    
    # Get unique image_id (should be same for all UIDs in group)
    image_ids = [uid_to_image_id(uid) for uid in group_uids]
    unique_image_id = image_ids[0]
    
    # Load the image once (all UIDs share the same image)
    image = decode_image_from_parquet(df_images, unique_image_id)
    
    # For this dataset, all items in a group share the same image
    # We'll display the same image with different QA annotations
    n = len(group_uids)
    rows = math.ceil(n / cols)
    
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, uid in enumerate(group_uids):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        # extract qa number for title
        _, qa_num = parse_uid(uid)
        plt.title(f"QA {qa_num}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def print_statistics(stats: dict) -> None:
    """
    Print dataset statistics.
    
    Args:
        stats: Statistics dictionary from group_by_image_id
    """
    print(f"\n{'='*60}")
    print(f"GUIAct Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total entries (UIDs in JSON): {stats['total_entries']}")
    print(f"Number of groups (unique image IDs): {stats['num_groups']}")
    print(f"Average QA pairs per group: {stats['avg_steps']:.2f}")
    print(f"Unmatched UID format rows: {stats['unmatched']}")
    print(f"{'='*60}\n")


def main_example():
    """
    Example usage of GUIAct data utilities.
    """
    # Download data
    json_path, parquet_path = download_guiact_files(split="train")
    
    # Group by image_id
    grouped_step_images, stats = group_by_image_id(json_path)
    
    # Print statistics
    print_statistics(stats)
    
    # Show example group
    print("\nExample group (first 1):")
    if len(grouped_step_images) > 0:
        first_group = grouped_step_images[0]
        print(first_group[:10], "..." if len(first_group) > 10 else "")
    
    # Load parquet
    df_images = pd.read_parquet(parquet_path)
    print(f"\nParquet info:")
    print(f"  Columns: {df_images.columns.tolist()}")
    print(f"  Index name: {df_images.index.name}")
    print(f"  Number of images: {len(df_images)}")
    
    # Visualize a group (requires matplotlib)
    if len(grouped_step_images) > 1:
        group_idx = 1
        visualize_group(
            group_uids=grouped_step_images[group_idx],
            df_images=df_images,
            group_idx=group_idx,
        )


if __name__ == "__main__":
    main_example()
