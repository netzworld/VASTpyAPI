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
    
    def connect(self, timeout=1000):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(timeout)
        self.client.connect((self.host, self.port))
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout)
            self.client.connect((self.host, self.port))
            return 1
        except Exception:
            # ensure client is None on failure
            try:
                if self.client:
                    self.client.close()
            except Exception:
                pass
            self.client = None
            return 0
    
    def disconnect(self):
        client = self.client
        if client is not None:
            try:
                client.close()
            except Exception as e:
                # log the exception or re-raise if you want callers to see it
                raise Exception("Error closing the socket connection", e)
                # pass
            finally:
                # avoid keeping a stale reference
                self.client = None
        return 1
       
    def send_command(self, command: str) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            payload = (command + "\n").encode("utf-8")
            sock.sendall(payload)
            response = sock.recv(4096)
            return response.decode("utf-8", errors="ignore")

# Client connecting to VAST Lite API
def main():
    vast = VASTControlClass("127.0.0.1", 22081)
    print(vast.connect())
    print(vast.disconnect())


if __name__ == "__main__":
    main()