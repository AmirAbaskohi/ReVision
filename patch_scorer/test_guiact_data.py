"""
Test script to verify GUIAct data loading.
Run this to verify the new data format works correctly.
"""

import sys
from pathlib import Path

# Add patch_scorer to path
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from patch_scorer.utils_guiact_data import (
    download_guiact_files,
    group_by_image_id,
    print_statistics,
    decode_image_from_parquet,
)
import pandas as pd


def test_data_download():
    """Test downloading and loading GUIAct data."""
    print("="*60)
    print("Testing GUIAct Data Download and Processing")
    print("="*60)
    
    # Download train data
    print("\n1. Downloading train data...")
    json_path, parquet_path = download_guiact_files(split="train")
    
    # Group by image_id
    print("\n2. Grouping data by image_id...")
    grouped_step_images, stats = group_by_image_id(json_path)
    
    # Print statistics
    print_statistics(stats)
    
    # Load parquet
    print("3. Loading image data from parquet...")
    df_images = pd.read_parquet(parquet_path)
    print(f"   ✓ Loaded {len(df_images)} images")
    print(f"   Columns: {df_images.columns.tolist()}")
    print(f"   Index: {df_images.index.name}")
    
    # Test image decoding
    print("\n4. Testing image decoding...")
    if len(grouped_step_images) > 0:
        first_group = grouped_step_images[0]
        first_uid = first_group[0]
        
        # Extract image_id from first uid
        from patch_scorer.utils_guiact_data import uid_to_image_id
        image_id = uid_to_image_id(first_uid)
        
        # Decode image
        image = decode_image_from_parquet(df_images, image_id)
        print(f"   ✓ Successfully decoded image")
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
    
    # Show example groups
    print("\n5. Example groups:")
    for i in range(min(3, len(grouped_step_images))):
        group = grouped_step_images[i]
        print(f"   Group {i}: {len(group)} QA pairs")
        print(f"   UIDs: {group[:3]}{'...' if len(group) > 3 else ''}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


def test_dataset_loading():
    """Test loading data with TemporalPatchScorerDataset."""
    print("\n" + "="*60)
    print("Testing TemporalPatchScorerDataset")
    print("="*60)
    
    try:
        from transformers import AutoProcessor
        from patch_scorer.temporal_dataset import TemporalPatchScorerDataset
        
        print("\n1. Loading processor...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("   ✓ Processor loaded")
        
        print("\n2. Creating dataset...")
        dataset = TemporalPatchScorerDataset(
            processor=processor,
            split="train",
            iou_threshold=0.5,
        )
        print(f"   ✓ Dataset created with {len(dataset)} samples")
        
        print("\n3. Loading first sample...")
        sample = dataset[0]
        print(f"   ✓ Sample loaded")
        print(f"   Keys: {list(sample.keys())}")
        print(f"   prev_pixel_values shape: {sample['prev_pixel_values'].shape}")
        print(f"   curr_pixel_values shape: {sample['curr_pixel_values'].shape}")
        print(f"   patch_scores_label shape: {sample['patch_scores_label'].shape}")
        
        print("\n" + "="*60)
        print("✓ Dataset loading test passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test basic data operations
    test_data_download()
    
    # Test dataset loading (optional, requires model download)
    try:
        test_dataset_loading()
    except Exception as e:
        print(f"\nSkipping dataset test (optional): {e}")
