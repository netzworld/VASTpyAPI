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

    anno = vast.get_anno_layer_nr_of_objects()
    print(f"Number of annotation objects: {anno}")
    an = vast.set_selected_anno_object_nr(1)
    print(f"Set selected annotation object to 1: {an}")
    
    vast.disconnect()

if __name__ == "__main__":
    main()