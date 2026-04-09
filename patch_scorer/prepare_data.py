"""
Data preparation script for Temporal Patch Scorer training.

This script:
1. Downloads GUIAct dataset (JSON + Parquet)
2. Groups consecutive QA pairs by image_id
3. Uses OmniParser to detect ALL UI elements in screenshots
4. Saves preprocessed data for training

Usage:
    python prepare_data.py --split train --output_dir ./data/processed
"""

import argparse
import base64
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, AutoModel


def parse_uid(uid: str) -> Tuple[str, str]:
    """
    Parse uid to extract image_id and qa_number.
    
    Format: uid_image_{image_id}_qa_{qa_num}
    """
    match = re.match(r'uid_image_([a-f0-9\-]+)_qa_(\d+)', uid)
    if match:
        image_id = match.group(1)
        qa_num = match.group(2)
        return image_id, qa_num
    raise ValueError(f"Invalid uid format: {uid}")


def download_guiact_data(split: str, cache_dir: str = None) -> Tuple[str, str]:
    """Download GUIAct JSON and Parquet files."""
    print(f"\n{'='*60}")
    print(f"Downloading GUIAct {split} split...")
    print(f"{'='*60}")
    
    repo_id = "yiye2023/GUIAct"
    json_filename = f"web-multi_{split}_data.json"
    parquet_filename = f"web-multi_{split}_images.parquet"
    
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=json_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    
    print(f"✓ JSON: {json_path}")
    print(f"✓ Parquet: {parquet_path}")
    
    return json_path, parquet_path


def load_guiact_data(json_path: str, parquet_path: str) -> Tuple[List[dict], pd.DataFrame]:
    """Load JSON metadata and image parquet."""
    print(f"\n{'='*60}")
    print("Loading GUIAct data...")
    print(f"{'='*60}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df_images = pd.read_parquet(parquet_path)
    
    print(f"✓ Loaded {len(data)} entries from JSON")
    print(f"✓ Loaded {len(df_images)} images from Parquet")
    
    return data, df_images


def group_by_image_id(data: List[dict]) -> Dict[str, List[dict]]:
    """Group QA pairs by image_id."""
    print(f"\n{'='*60}")
    print("Grouping by image_id...")
    print(f"{'='*60}")
    
    groups = defaultdict(list)
    skipped = 0
    
    for entry in data:
        uid = entry.get('uid', '')
        try:
            image_id, qa_num = parse_uid(uid)
            groups[image_id].append({
                'qa_num': qa_num,
                'uid': uid,
                'entry': entry,
            })
        except ValueError:
            skipped += 1
            continue
    
    # Sort by qa_num within each group
    for image_id in groups:
        groups[image_id].sort(key=lambda x: x['qa_num'])
    
    print(f"✓ Created {len(groups)} image groups")
    print(f"✓ Skipped {skipped} entries with invalid UIDs")
    
    return groups


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


def load_omniparser_model(device: str = "cuda"):
    """Load OmniParser model for UI element detection."""
    print(f"\n{'='*60}")
    print("Loading OmniParser model...")
    print(f"{'='*60}")
    
    model_name = "microsoft/OmniParser-v2.0"
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    print(f"✓ OmniParser loaded on {device}")
    
    return processor, model


def parse_ui_elements_with_omniparser(
    image: Image.Image,
    processor,
    model,
    device: str = "cuda"
) -> List[Dict]:
    """
    Use OmniParser to detect all UI elements in screenshot.
    
    Returns:
        List of component dicts with 'bbox' field [x1, y1, x2, y2]
    """
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Parse outputs to extract bboxes
    # Note: This needs to be adapted based on OmniParser's actual output format
    # The exact format may vary - check OmniParser documentation
    components = []
    
    if hasattr(outputs, 'pred_boxes') and hasattr(outputs, 'pred_labels'):
        boxes = outputs.pred_boxes[0].cpu().numpy()  # [N, 4]
        labels = outputs.pred_labels[0].cpu().numpy()  # [N]
        scores = outputs.scores[0].cpu().numpy() if hasattr(outputs, 'scores') else None
        
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            # Filter by confidence if available
            if scores is not None and scores[idx] < 0.3:
                continue
            
            # Convert normalized coords to absolute if needed
            if box.max() <= 1.0:
                x1, y1, x2, y2 = box
                x1 = int(x1 * image.width)
                y1 = int(y1 * image.height)
                x2 = int(x2 * image.width)
                y2 = int(y2 * image.height)
            else:
                x1, y1, x2, y2 = map(int, box)
            
            components.append({
                'id': idx,
                'bbox': [x1, y1, x2, y2],
                'label': int(label),
                'score': float(scores[idx]) if scores is not None else 1.0,
            })
    
    return components


def create_temporal_pairs(groups: Dict[str, List[dict]], df_images: pd.DataFrame) -> List[Dict]:
    """Create consecutive temporal pairs from grouped data."""
    print(f"\n{'='*60}")
    print("Creating temporal pairs...")
    print(f"{'='*60}")
    
    temporal_pairs = []
    
    for image_id, group_items in groups.items():
        # Create consecutive pairs within the group
        for i in range(len(group_items) - 1):
            prev_item = group_items[i]
            curr_item = group_items[i + 1]
            
            temporal_pairs.append({
                'image_id': image_id,
                'prev_qa_num': prev_item['qa_num'],
                'curr_qa_num': curr_item['qa_num'],
                'prev_uid': prev_item['uid'],
                'curr_uid': curr_item['uid'],
                'prev_entry': prev_item['entry'],
                'curr_entry': curr_item['entry'],
            })
    
    print(f"✓ Created {len(temporal_pairs)} temporal pairs")
    
    return temporal_pairs


def process_and_save_data(
    temporal_pairs: List[Dict],
    df_images: pd.DataFrame,
    processor,
    model,
    output_dir: Path,
    device: str = "cuda"
):
    """Process all temporal pairs and save to disk."""
    print(f"\n{'='*60}")
    print("Processing temporal pairs with OmniParser...")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data = []
    cache_elements = {}  # Cache elements by image_id
    
    for pair_idx, pair in enumerate(tqdm(temporal_pairs, desc="Processing pairs")):
        image_id = pair['image_id']
        
        # Get image from parquet (same image for both prev and curr in this dataset)
        if image_id not in cache_elements:
            # Load image
            if image_id in df_images.index:
                base64_str = df_images.loc[image_id, 'base64']
            else:
                row = df_images[df_images.index == image_id].iloc[0]
                base64_str = row['base64']
            
            image = decode_base64_image(base64_str)
            
            # Parse UI elements with OmniParser
            try:
                elements = parse_ui_elements_with_omniparser(image, processor, model, device)
            except Exception as e:
                print(f"\nWarning: Failed to parse image {image_id}: {e}")
                elements = []
            
            # Cache the elements
            cache_elements[image_id] = {
                'elements': elements,
                'image_size': {'width': image.width, 'height': image.height},
                'base64': base64_str,
            }
        
        # Get cached data
        image_data = cache_elements[image_id]
        
        # For temporal pairs, we use the same elements for both prev and curr
        # since they share the same image_id (same screenshot)
        processed_pair = {
            'pair_id': f"pair_{pair_idx:06d}",
            'image_id': image_id,
            'prev_qa_num': pair['prev_qa_num'],
            'curr_qa_num': pair['curr_qa_num'],
            'prev_uid': pair['prev_uid'],
            'curr_uid': pair['curr_uid'],
            'image_size': image_data['image_size'],
            'base64': image_data['base64'],
            'elements': image_data['elements'],  # All UI elements from OmniParser
            'prev_action': pair['prev_entry'].get('actions_label', []),
            'curr_action': pair['curr_entry'].get('actions_label', []),
            'prev_question': pair['prev_entry'].get('question', ''),
            'curr_question': pair['curr_entry'].get('question', ''),
        }
        
        processed_data.append(processed_pair)
    
    # Save processed data
    output_file = output_dir / "temporal_pairs_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n✓ Saved {len(processed_data)} processed pairs to: {output_file}")
    
    # Save statistics
    stats = {
        'total_pairs': len(processed_data),
        'unique_images': len(cache_elements),
        'avg_elements_per_image': sum(len(v['elements']) for v in cache_elements.values()) / len(cache_elements) if cache_elements else 0,
    }
    
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to: {stats_file}")
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Unique images: {stats['unique_images']}")
    print(f"  Avg elements per image: {stats['avg_elements_per_image']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GUIAct data for temporal patch scorer training")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to process")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for HuggingFace downloads")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for OmniParser inference")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.split
    
    print(f"\n{'='*60}")
    print(f"GUIAct Data Preparation Pipeline")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    
    # Step 1: Download GUIAct data
    json_path, parquet_path = download_guiact_data(args.split, args.cache_dir)
    
    # Step 2: Load data
    data, df_images = load_guiact_data(json_path, parquet_path)
    
    # Step 3: Group by image_id
    groups = group_by_image_id(data)
    
    # Step 4: Create temporal pairs
    temporal_pairs = create_temporal_pairs(groups, df_images)
    
    # Step 5: Load OmniParser
    processor, model = load_omniparser_model(args.device)
    
    # Step 6: Process all pairs and save
    process_and_save_data(
        temporal_pairs=temporal_pairs,
        df_images=df_images,
        processor=processor,
        model=model,
        output_dir=output_dir,
        device=args.device
    )
    
    print(f"\n{'='*60}")
    print("✓ Data preparation complete!")
    print(f"{'='*60}")
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the processed data")
    print(f"  2. Run training: python train_temporal_scorer.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
