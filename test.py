from VASTControlClass2 import VASTControlClass
import numpy as np

def run_test():
    # 1. Initialize the class
    vast = VASTControlClass()
    
    # Connection parameters (Default VAST port is usually 22081)
    HOST = '127.0.0.1'
    PORT = 22081
    TIMEOUT = 1000  # milliseconds

    print(f"Attempting to connect to VAST at {HOST}:{PORT}...")
    
    # 2. Connect
    res = vast.connect(HOST, PORT, TIMEOUT)
    
    if res == 0:
        print(" Connection Failed.")
        print("   - Ensure VAST Lite is running.")
        print("   - Ensure 'Remote Control Server' is enabled in VAST settings.")
        print(f"   - Error Code: {vast.getlasterror()}")
        return

    print(" Connected to VAST!")

    try:
        # 3. Test: Get Info (General Metadata)
        print("\n--- Testing getinfo() ---")
        info, res = vast.getinfo()
        if res == 1 and info is not None:
            print(" Info received:")
            for k, v in info.items():
                print(f"   {k}: {v}")
        else:
            print(f" Failed to get info. Error: {vast.getlasterror()}")

        # 4. Test: Get Layer Info (Check Layer 1)
        print("\n--- Testing getlayerinfo(1) ---")
        layer_info, res = vast.getlayerinfo(1)
        if res == 1 and layer_info is not None:
            print(" Layer 1 Info received:")
            for k, v in layer_info.items():
                print(f"   {k}: {v}")
        else:
            print(f"️ Layer 1 not found or error (Error: {vast.getlasterror()})")

        # 5. Test: Get Segment Data (Check Segment ID 1)
        print("\n--- Testing getsegmentdata(1) ---")
        seg_data, res = vast.getsegmentdata(1)
        if res == 1 and seg_data is not None:
            print(" Segment 1 Data received:")
            print(f"   Anchor Point: {seg_data['anchorpoint']}")
            print(f"   Bounding Box: {seg_data['boundingbox']}")
        else:
            print(f"️ Segment 1 not found (Error: {vast.getlasterror()})")

        # 6. Test: Get Raw Image Data (Small 64x64x64 cube)
        print("\n--- Testing getsegimageraw() ---")
        # Parameters: miplevel, minx, maxx, miny, maxy, minz, maxz
        # Requesting a 64^3 block at mip 0
        img, res = vast.getsegimageraw(0, 0, 63, 0, 63, 0, 63)
        
        if res == 1 and img is not None:
            print(f" Image Data received successfully!")
            print(f"   Shape: {img.shape} (Expected: (64, 64, 64))")
            print(f"   Data Type: {img.dtype}")
            print(f"   Non-zero voxels: {np.count_nonzero(img)}")
        else:
            print(f" Failed to get image data. Error: {vast.getlasterror()}")

    except Exception as e:
        print(f"\n Exception occurred during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7. Disconnect
        print("\n--- Disconnecting ---")
        vast.disconnect()
        print("Done.")

if __name__ == "__main__":
    run_test()