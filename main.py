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
    n = vast.get_number_of_layers()
    print("layers:", n)

    vast.disconnect()

if __name__ == "__main__":
    main()