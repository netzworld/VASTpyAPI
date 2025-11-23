from VASTControlClass import VASTControlClass

def main():
    vast = VASTControlClass() # Defaults to 127.0.0.1:22081
    if not vast.connect():
        print("Failed to connect to VAST.")
        return
    # print("api version: ", vast.get_api_version())
    # info = vast.get_info()
    # print(f"info is a {type(info[0])} as: {info[0]}")
    # hw_info = vast.get_hardware_info()
    # print(f"hw_info is a {type(hw_info[0])} as: {hw_info[0]}")
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect to VAST")
        exit(1)

    print("Connected successfully")

    # Test basic info
    info = vast.get_info()
    if info:
        print(f"Dataset size: {info['datasizex']}x{info['datasizey']}x{info['datasizez']}")
        print(f"Voxel size: {info['voxelsizex']}, {info['voxelsizey']}, {info['voxelsizez']} nm")
    else:
        print("No dataset loaded")

    # Test API version
    version = vast.get_api_version()
    print(f"API version: {version}")

    # Test layer info
    num_layers = vast.get_number_of_layers()
    print(f"Number of layers: {num_layers}")

    if num_layers > 0:
        layer_info = vast.get_layer_info(0)
        if layer_info:
            print(f"Layer 0: {layer_info['name']} ({layer_info['type']})")

    # Test segmentation
    num_segs = vast.get_number_of_segments()
    print(f"Number of segments: {num_segs}")

    if num_segs > 0:
        # Get first segment data
        seg = vast.get_segment_data(1)
        if seg:
            print(f"Segment 1 bounding box: {seg['boundingbox']}")
            
            # Try getting a small image region
            bbox = seg['boundingbox']
            img = vast.get_seg_image_rle_decoded(
                miplevel=0,
                minx=bbox[0], maxx=min(bbox[3], bbox[0]+50),
                miny=bbox[1], maxy=min(bbox[4], bbox[1]+50),
                minz=bbox[2], maxz=bbox[2]  # Single slice
            )
            
            if img is not None:
                print(f"Retrieved image shape: {img.shape}")
                print(f"Unique segment IDs in region: {len(set(img.flatten()))}")



    vast.disconnect()

if __name__ == "__main__":
    main()