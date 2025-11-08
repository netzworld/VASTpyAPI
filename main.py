import socket
import struct
from typing import Tuple, Optional, Dict, Any, List
import re

HOST = "127.0.0.1"
PORT = 22081

errorCodes = {
    0: "No error",
    1: "Unknown error",
    2: "Unexpected data received from VAST - API mismatch?",
    3: "VAST received invalid data. Command ignored.",
    4: "VAST internal data read failure",
    5: "Internal VAST error",
    6: "Could not complete command because modifying the view in VAST is disabled",
    7: "Could not complete command because modifying the segmentation in VAST is disabled",
    10: "Coordinates out of bounds",
    11: "Mip level out of bounds",
    12: "Data size overflow",
    13: "Data size mismatch - Specified coordinates and data block size do not match",
    14: "Parameter out of bounds",
    15: "Could not enable diskcache (please set correct folder in VAST Preferences)",
    20: "Annotation object or segment number out of bounds",
    21: "No annotation or segmentation available",
    22: "RLE overflow - RLE-encoding makes the data larger than raw; please request as raw",
    23: "Invalid annotation object type",
    24: "Annotation operation failed",
    30: "Invalid layer number",
    31: "Invalid layer type",
    32: "Layer operation failed",
    50: "sourcearray and targetarray must have the same length",
}

class VASTControlClass:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        # runtime socket client (None until connected)
        self.client: Optional[socket.socket] = None
    
    def connect(self, timeout=100):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout)
            self.client.connect((self.host, self.port))
            return 1
        except Exception as e:
            print(f"Connection failed: {e}")
            try:
                if self.client:
                    self.client.close()
            except Exception:
                pass
            self.client = None
            return 0
    
    def disconnect(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
        return 1
        
    def send_command(self, msg_id: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """
        Send a binary command to the VAST API and return the message type and response bytes.

        Each message follows the format:
        [0..3]   = b"VAST"
        [4..11]  = uint64 length of data following (payload + 4 bytes for msg_id)
        [12..15] = uint32 message ID (little-endian)
        [16..]   = payload (optional)
        """
        total_len = len(payload) + 4
        header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
        message = header + payload

        print(f"Sending {len(message)} bytes: {message.hex()}")

        client = self.client
        if client is None:
            raise RuntimeError("Not connected to VAST API.")
        client.sendall(message)

        hdr = client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr!r}")

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type = struct.unpack("<I", hdr[12:16])[0]

        # Receive payload 
        payload_bytes = b""
        while len(payload_bytes) < total_len - 4:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload_bytes += chunk

        # print(f"Response msg_type={msg_type}, len={len(payload_bytes)}")
        return msg_type, payload_bytes


    # [info, res] = getinfo()
    # Reads out general information from VAST.
    # Returns a struct with the following fields if successful, or an empty struct [] if failed:
    def get_info(self):
        msg_type, data = self.send_command(6)  # GETINFO = 6
        if msg_type == 21:  # error
            print("No segmentation or dataset loaded in VAST.")
            return None
        return data

# Client connecting to VAST Lite API
def main():
    vast = VASTControlClass(HOST,PORT)
    vast.connect()
    print(vast.get_info())
    vast.disconnect()

if __name__ == "__main__":
    main()