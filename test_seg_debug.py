"""
Debug script to test segmentation data loading directly.
"""
from VASTControlClass import VASTControlClass
import numpy as np

def test_seg_loading():
    print("=" * 60)
    print("Testing Segmentation Data Loading")
    print("=" * 60)
    
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect to VAST")
        return
    
    print("\n[1] Getting all segment data matrix...")
    data, res = vast.get_all_segment_data_matrix()
    if res and data is not None:
        data = np.array(data)
        print(f"   Got {len(data)} segments")
        print(f"   Segment IDs: {data[:5, 0] if len(data) > 0 else 'none'}...")
    else:
        print("   FAIL: No segment data")
        return
    
    print("\n[2] Getting selected layer info...")
    layers = vast.get_selected_layer_nr()
    print(f"   Layers: {layers}")
    
    seg_layer = layers.get('selected_segment_layer', -1) if layers else -1
    print(f"   Segment layer: {seg_layer}")
    
    print("\n[3] Testing get_seg_image_rle_decoded_bboxes...")
    # Test with a small region
    mip = 2
    xmin, xmax = 250, 280
    ymin, ymax = 1000, 1030
    zmin, zmax = 0, 5
    
    print(f"   Region: mip={mip}, x=[{xmin}-{xmax}], y=[{ymin}-{ymax}], z=[{zmin}-{zmax}]")
    
    result = vast.get_seg_image_rle_decoded_bboxes(mip, xmin, xmax, ymin, ymax, zmin, zmax, False)
    
    if result is None:
        print("   FAIL: get_seg_image_rle_decoded_bboxes returned None")
        return
    
    print(f"   Return type: {type(result)}")
    print(f"   Return length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    
    if isinstance(result, tuple) and len(result) >= 4:
        seg_image, values, numbers, bboxes = result
        print(f"\n   seg_image shape: {seg_image.shape if hasattr(seg_image, 'shape') else type(seg_image)}")
        print(f"   seg_image dtype: {seg_image.dtype if hasattr(seg_image, 'dtype') else 'N/A'}")
        print(f"   seg_image unique values: {np.unique(seg_image)[:10] if hasattr(seg_image, 'shape') else 'N/A'}...")
        print(f"   values: {values[:10] if values is not None and len(values) > 0 else 'empty/None'}")
        print(f"   numbers: {numbers[:10] if numbers is not None and len(numbers) > 0 else 'empty/None'}")
        print(f"   bboxes shape: {bboxes.shape if hasattr(bboxes, 'shape') else type(bboxes)}")
    else:
        print(f"   Raw result: {result}")
    
    print("\n[4] Testing with mip=0 (no scaling)...")
    # Larger coordinates at mip 0
    xmin0, xmax0 = 1000, 1030
    ymin0, ymax0 = 4000, 4030
    zmin0, zmax0 = 0, 5
    
    print(f"   Region: mip=0, x=[{xmin0}-{xmax0}], y=[{ymin0}-{ymax0}], z=[{zmin0}-{zmax0}]")

    result = vast.get_seg_image_rle_decoded_bboxes(0, xmin0, xmax0, ymin0, ymax0, zmin0, zmax0, False)
    
    if result is None:
        print("   FAIL: returned None")
    elif isinstance(result, tuple) and len(result) >= 4:
        seg_image, values, numbers, bboxes = result
        print(f"   seg_image shape: {seg_image.shape if hasattr(seg_image, 'shape') else type(seg_image)}")
        print(f"   seg_image unique: {np.unique(seg_image)[:10]}...")
        print(f"   values: {values[:10] if values is not None and len(values) > 0 else 'empty/None'}")
    else:
        print(f"   Raw result: {result}")
    
    vast.disconnect()
    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)

if __name__ == "__main__":
    test_seg_loading()
