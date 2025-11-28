"""
test_main.py - Tests for VAST segmentation image retrieval functions
"""

import numpy as np
from VASTControlClass import VASTControlClass

def test_get_seg_image_raw():
    """Test raw segmentation image retrieval."""
    print("\n=== Testing get_seg_image_raw ===")
    
    vast = VASTControlClass()
    if not vast.connect():
        print("❌ Failed to connect")
        return False
    
    try:
        # Get dataset info
        info = vast.get_info()
        if not info:
            print("❌ No dataset loaded")
            return False
        
        print(f"✓ Dataset: {info['datasizex']}x{info['datasizey']}x{info['datasizez']}")
        
        # Get a small region at mip level 2 (lower resolution)
        miplevel = 2
        minx, miny, minz = 0, 0, 0
        maxx, maxy, maxz = 99, 99, 0  # 100x100x1 region
        
        print(f"Fetching raw image at mip {miplevel}: ({minx},{miny},{minz}) to ({maxx},{maxy},{maxz})")
        
        raw_data = vast.get_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz)
        
        if raw_data is None:
            print(f"❌ Failed to get raw image. Error: {vast.get_last_error()}")
            return False
        
        # Convert to uint16 array
        seg_array = np.frombuffer(raw_data, dtype=np.uint16)
        expected_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
        
        print(f"✓ Received {len(seg_array)} values (expected {expected_size})")
        print(f"✓ Data type: uint16")
        print(f"✓ Unique segment IDs: {len(np.unique(seg_array))}")
        print(f"✓ Value range: {seg_array.min()} to {seg_array.max()}")
        
        return True
        
    finally:
        vast.disconnect()


def test_get_seg_image_rle():
    """Test RLE-encoded segmentation image retrieval."""
    print("\n=== Testing get_seg_image_rle ===")
    
    vast = VASTControlClass()
    if not vast.connect():
        print("❌ Failed to connect")
        return False
    
    try:
        info = vast.get_info()
        if not info:
            print("❌ No dataset loaded")
            return False
        
        # Get same region as raw test
        miplevel = 2
        minx, miny, minz = 0, 0, 0
        maxx, maxy, maxz = 99, 99, 0
        
        print(f"Fetching RLE image at mip {miplevel}: ({minx},{miny},{minz}) to ({maxx},{maxy},{maxz})")
        
        rle_data = vast.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz)
        
        if rle_data is None:
            print(f"❌ Failed to get RLE image. Error: {vast.get_last_error()}")
            return False
        
        # RLE data is pairs of (value, count)
        rle_array = np.frombuffer(rle_data, dtype=np.uint16)
        
        print(f"✓ Received {len(rle_array)} RLE pairs ({len(rle_array)//2} runs)")
        
        # Verify RLE format (should be even number of values)
        if len(rle_array) % 2 != 0:
            print("❌ Invalid RLE data (odd number of values)")
            return False
        
        # Calculate total pixels from RLE
        total_pixels = 0
        for i in range(0, len(rle_array), 2):
            count = rle_array[i + 1]
            total_pixels += count
        
        expected_size = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1)
        print(f"✓ RLE decodes to {total_pixels} pixels (expected {expected_size})")
        
        if total_pixels != expected_size:
            print(f"⚠️  Warning: Size mismatch!")
            return False
        
        # Check compression ratio
        raw_size = expected_size * 2  # uint16 = 2 bytes
        rle_size = len(rle_data)
        compression_ratio = raw_size / rle_size
        print(f"✓ Compression ratio: {compression_ratio:.2f}x")
        
        return True
        
    finally:
        vast.disconnect()


def test_get_seg_image_rle_decoded():
    """Test RLE-decoded segmentation image retrieval."""
    print("\n=== Testing get_seg_image_rle_decoded ===")
    
    vast = VASTControlClass()
    if not vast.connect():
        print("❌ Failed to connect")
        return False
    
    try:
        info = vast.get_info()
        if not info:
            print("❌ No dataset loaded")
            return False
        
        # Test with a small 3D region
        miplevel = 2
        minx, miny, minz = 0, 0, 0
        maxx, maxy, maxz = 49, 49, 2  # 50x50x3 region
        
        print(f"Fetching decoded RLE at mip {miplevel}: ({minx},{miny},{minz}) to ({maxx},{maxy},{maxz})")
        
        seg_volume = vast.get_seg_image_rle_decoded(
            miplevel, minx, maxx, miny, maxy, minz, maxz
        )
        
        if seg_volume is None:
            print(f"❌ Failed to get decoded image. Error: {vast.get_last_error()}")
            return False
        
        expected_shape = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
        print(f"✓ Shape: {seg_volume.shape} (expected {expected_shape})")
        print(f"✓ Data type: {seg_volume.dtype}")
        
        if seg_volume.shape != expected_shape:
            print("❌ Shape mismatch!")
            return False
        
        # Analyze content
        unique_ids = np.unique(seg_volume)
        print(f"✓ Unique segment IDs: {len(unique_ids)}")
        print(f"✓ Segment IDs present: {unique_ids[:10]}{'...' if len(unique_ids) > 10 else ''}")
        print(f"✓ Value range: {seg_volume.min()} to {seg_volume.max()}")
        
        # Check for background (ID 0)
        background_voxels = np.sum(seg_volume == 0)
        total_voxels = seg_volume.size
        print(f"✓ Background voxels: {background_voxels}/{total_voxels} ({100*background_voxels/total_voxels:.1f}%)")
        
        return True
        
    finally:
        vast.disconnect()


def test_set_seg_translation():
    """Test segment ID translation during retrieval."""
    print("\n=== Testing set_seg_translation ===")
    
    vast = VASTControlClass()
    if not vast.connect():
        print("❌ Failed to connect")
        return False
    
    try:
        # First get image without translation
        miplevel = 2
        minx, miny, minz = 0, 0, 0
        maxx, maxy, maxz = 99, 99, 0
        
        print("Fetching image WITHOUT translation...")
        original = vast.get_seg_image_rle_decoded(
            miplevel, minx, maxx, miny, maxy, minz, maxz
        )
        
        if original is None:
            print("❌ Failed to get original image")
            return False
        
        original_ids = np.unique(original)
        print(f"✓ Original unique IDs: {original_ids[:10]}")
        
        # Set up translation: map first few segments to new IDs
        # Everything not in source_array becomes 0
        if len(original_ids) < 2:
            print("⚠️  Not enough segments to test translation")
            return True
        
        source_array = [int(original_ids[1])]  # Skip background (0)
        target_array = [999]  # Map to ID 999
        
        print(f"Setting translation: {source_array} -> {target_array}")
        
        if not vast.set_seg_translation(source_array, target_array):
            print(f"❌ Failed to set translation. Error: {vast.get_last_error()}")
            return False
        
        print("✓ Translation set successfully")
        
        # Get image WITH translation
        print("Fetching image WITH translation...")
        translated = vast.get_seg_image_rle_decoded(
            miplevel, minx, maxx, miny, maxy, minz, maxz
        )
        
        if translated is None:
            print("❌ Failed to get translated image")
            return False
        
        translated_ids = np.unique(translated)
        print(f"✓ Translated unique IDs: {translated_ids}")
        
        # Verify translation worked
        if 999 in translated_ids:
            print("✓ Translation successful: ID 999 present")
            count_999 = np.sum(translated == 999)
            count_original = np.sum(original == source_array[0])
            print(f"✓ Voxel count match: {count_999} vs {count_original}")
        else:
            print("❌ Translation failed: ID 999 not found")
            return False
        
        # Clear translation
        print("Clearing translation...")
        if not vast.set_seg_translation([], []):
            print("❌ Failed to clear translation")
            return False
        
        print("✓ Translation cleared")
        
        return True
        
    finally:
        vast.disconnect()


def test_compare_raw_vs_rle():
    """Compare raw and RLE retrieval to ensure they match."""
    print("\n=== Testing RAW vs RLE comparison ===")
    
    vast = VASTControlClass()
    if not vast.connect():
        print("❌ Failed to connect")
        return False
    
    try:
        miplevel = 2
        minx, miny, minz = 100, 100, 0
        maxx, maxy, maxz = 149, 149, 0
        
        print(f"Fetching region: ({minx},{miny},{minz}) to ({maxx},{maxy},{maxz})")
        
        # Get raw
        raw_data = vast.get_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz)
        if raw_data is None:
            print("❌ Failed to get raw data")
            return False
        
        raw_array = np.frombuffer(raw_data, dtype=np.uint16)
        
        # Get RLE decoded
        rle_decoded = vast.get_seg_image_rle_decoded(
            miplevel, minx, maxx, miny, maxy, minz, maxz
        )
        if rle_decoded is None:
            print("❌ Failed to get RLE decoded data")
            return False
        
        # Raw is in X-major order, RLE decoded is in Y-major order (Y, X, Z)
        # So we need to reshape raw and compare
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        
        raw_reshaped = raw_array.reshape(size_x, size_y, size_z)
        raw_transposed = np.transpose(raw_reshaped, (1, 0, 2))
        
        # Compare
        if np.array_equal(raw_transposed, rle_decoded):
            print("✓ RAW and RLE decoded data match perfectly!")
            return True
        else:
            differences = np.sum(raw_transposed != rle_decoded)
            total = raw_transposed.size
            print(f"❌ Mismatch: {differences}/{total} voxels differ ({100*differences/total:.2f}%)")
            return False
        
    finally:
        vast.disconnect()


def run_all_tests():
    """Run all segmentation image tests."""
    print("=" * 60)
    print("VAST Segmentation Image Retrieval Tests")
    print("=" * 60)
    
    tests = [
        ("Raw Image Retrieval", test_get_seg_image_raw),
        ("RLE Image Retrieval", test_get_seg_image_rle),
        ("RLE Decoded Retrieval", test_get_seg_image_rle_decoded),
        ("Segment Translation", test_set_seg_translation),
        ("RAW vs RLE Comparison", test_compare_raw_vs_rle),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Exception in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)