"""
Test script for surface extraction functionality.
Run this with VAST open and a dataset with segmentation loaded.
"""
import os
from exportSkeleton import SurfaceExtractor
from VASTControlClass import VASTControlClass

def test_surface_extraction():
    print("=" * 60)
    print("VAST Surface Extraction Test")
    print("=" * 60)
    
    # Configure export parameters for a quick test
    export_params = {
        'miplevel': 1,           # Use mip level 2 for faster testing
        'extractwhich': 1,       # Selected segment and children, uncollapsed
        'blocksizex': 128,
        'blocksizey': 128,
        'blocksizez': 64,
        'overlap': 2,
        'slicestep': 1,
        'xunit': 5.0,            # Voxel size in nm
        'yunit': 5.0,
        'zunit': 50.0,
        'xscale': 0.001,
        'yscale': 0.001,
        'zscale': 0.001,
        'targetfolder': './test_output_meshes/',
        'targetfileprefix': 'test_',
        'fileformat': 1,         # 1 = OBJ
        'closesurfaces': 1,
        'invertz': 0,
        'includefoldernames': 0,
        'disablenetwarnings': 1,
        'erodedilate': 0,
        'usemipregionconstraint': 0,
        'outputoffsetx': 0,
        'outputoffsety': 0, 
        'outputoffsetz': 0,
        'objectcolors': 1,       # Use VAST colors
    }
    
    # Small region for testing
    # Note: At mip level 2, these coordinates will be divided by 4
    # So these full-res coordinates become [250-377, 250-377, 0-20] at mip 2
    region_params = {
        'xmin': 0,
        'xmax': 2713,
        'ymin': 0,
        'ymax': 2713,
        'zmin': 0,
        'zmax': 20,
    }
    
    # Create output directory
    os.makedirs(export_params['targetfolder'], exist_ok=True)
    
    print(f"\nExport parameters:")
    print(f"  MIP level: {export_params['miplevel']}")
    print(f"  Extract mode: {export_params['extractwhich']}")
    print(f"  Block size: {export_params['blocksizex']}x{export_params['blocksizey']}x{export_params['blocksizez']}")
    print(f"  Region: X[{region_params['xmin']}-{region_params['xmax']}] "
          f"Y[{region_params['ymin']}-{region_params['ymax']}] "
          f"Z[{region_params['zmin']}-{region_params['zmax']}]")
    print(f"  Output folder: {export_params['targetfolder']}")

    print("\n[1] Connecting to VAST...")
    try:
        vast = VASTControlClass()
        if not vast.connect():
            print("   FAIL: Could not connect to VAST at 127.0.0.1:22081")
            print("   Make sure VAST is running and dataset is loaded")
            return
        print("   PASS: Connected to VAST")
    except Exception as e:
        print(f"   FAIL: Connection error: {e}")
        return

    print("\n[2] Initializing SurfaceExtractor...")
    try:
        extractor = SurfaceExtractor(vast, export_params, region_params)
        print("   PASS: SurfaceExtractor created")
    except Exception as e:
        print(f"   FAIL: {e}")
        vast.disconnect()
        return

    print("\n[3] Running extract_surfaces()...")
    try:
        extractor.extract_surfaces()
        print("   PASS: extract_surfaces() completed")
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        vast.disconnect()
        return

    print("\n[4] Running export_meshes()...")
    try:
        extractor.export_meshes()
        print("   PASS: export_meshes() completed")
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        vast.disconnect()
        return

    print("\n[5] Checking output files...")
    output_files = os.listdir(export_params['targetfolder'])
    obj_files = [f for f in output_files if f.endswith('.obj')]
    mtl_files = [f for f in output_files if f.endswith('.mtl')]
    
    print(f"   Found {len(obj_files)} OBJ files, {len(mtl_files)} MTL files")
    if obj_files:
        print(f"   Sample files: {obj_files[:3]}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    # Disconnect from VAST
    print("\nDisconnecting from VAST...")
    vast.disconnect()
    print("Done!")

if __name__ == "__main__":
    test_surface_extraction()
