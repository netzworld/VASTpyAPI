import socket
import struct
from typing import Tuple, Optional, Dict, Any, List
import re

HOST = "127.0.0.1"
PORT = 22081

# VAST API message IDs
GETINFO = 1;
GETNUMBEROFSEGMENTS = 2;
GETSEGMENTDATA     = 3;
GETSEGMENTNAME     = 4;
SETANCHORPOINT     = 5;
SETSEGMENTNAME     = 6;
SETSEGMENTCOLOR    = 7;
GETVIEWCOORDINATES = 8;
GETVIEWZOOM        = 9;
SETVIEWCOORDINATES = 10;
SETVIEWZOOM        = 11;
GETNROFLAYERS      = 12;
GETLAYERINFO       = 13;
GETALLSEGMENTDATA  = 14;
GETALLSEGMENTNAMES = 15;
SETSELECTEDSEGMENTNR = 16;
GETSELECTEDSEGMENTNR = 17;
SETSELECTEDLAYERNR = 18;
GETSELECTEDLAYERNR = 19;
GETSEGIMAGERAW     = 20;
GETSEGIMAGERLE     = 21;
GETSEGIMAGESURFRLE = 22;
SETSEGTRANSLATION  = 23;
GETSEGIMAGERAWIMMEDIATE = 24;
GETSEGIMAGERLEIMMEDIATE = 25;
GETEMIMAGERAW      = 30;
GETEMIMAGERAWIMMEDIATE = 31;
REFRESHLAYERREGION = 32;
GETPIXELVALUE      = 33;
GETSCREENSHOTIMAGERAW = 40;
GETSCREENSHOTIMAGERLE = 41;
SETSEGIMAGERAW     = 50;
SETSEGIMAGERLE     = 51;
SETSEGMENTBBOX     = 60;
GETFIRSTSEGMENTNR  = 61;
GETHARDWAREINFO    = 62;
ADDSEGMENT         = 63;
MOVESEGMENT        = 64;
GETDRAWINGPROPERTIES = 65;
SETDRAWINGPROPERTIES = 66;
GETFILLINGPROPERTIES = 67;
SETFILLINGPROPERTIES = 68;

GETANNOLAYERNROFOBJECTS = 70;
GETANNOLAYEROBJECTDATA  = 71;
GETANNOLAYEROBJECTNAMES  = 72;
ADDNEWANNOOBJECT = 73;
MOVEANNOOBJECT = 74;
REMOVEANNOOBJECT = 75;
SETSELECTEDANNOOBJECTNR = 76;
GETSELECTEDANNOOBJECTNR = 77;
GETAONODEDATA = 78;
GETAONODELABELS = 79;

SETSELECTEDAONODEBYDFSNR = 80;
SETSELECTEDAONODEBYCOORDS = 81;
GETSELECTEDAONODENR = 82;
ADDAONODE           = 83;
MOVESELECTEDAONODE          = 84;
REMOVESELECTEDAONODE        = 85;
SWAPSELECTEDAONODECHILDREN  = 86;
MAKESELECTEDAONODEROOT      = 87;

SPLITSELECTEDSKELETON = 88;
WELDSKELETONS = 89;

GETANNOOBJECT = 90;
SETANNOOBJECT = 91;
ADDANNOOBJECT = 92;
GETCLOSESTAONODEBYCOORDS = 93;
GETAONODEPARAMS = 94;
SETAONODEPARAMS = 95;

GETAPIVERSION      = 100;
GETAPILAYERSENABLED = 101;
SETAPILAYERSENABLED = 102;
GETSELECTEDAPILAYERNR = 103;
SETSELECTEDAPILAYERNR = 104;

GETCURRENTUISTATE  = 110;
GETERRORPOPUPSENABLED = 112;
SETERRORPOPUPSENABLED = 113;
SETUIMODE          = 114;
SHOWWINDOW         = 115;
SET2DVIEWORIENTATION = 116;
GETPOPUPSENABLED = 117;
SETPOPUPSENABLED = 118;

ADDNEWLAYER        = 120;
LOADLAYER          = 121;
SAVELAYER          = 122;
REMOVELAYER        = 123;
MOVELAYER          = 124;
SETLAYERINFO       = 125;
GETMIPMAPSCALEFACTORS = 126;

EXECUTEFILL        = 131;
EXECUTELIMITEDFILL = 132;
EXECUTECANVASPAINTSTROKE = 133;
EXECUTESTARTAUTOSKELETONIZATION=134;
EXECUTESTOPAUTOSKELETONIZATION=135;
EXECUTEISAUTOSKELETONIZATIONDONE=136;

SETTOOLPARAMETERS  = 151;

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
        self.host                            = host
        self.port                            = port
        self.client: Optional[socket.socket] = None
        self.last_error                      = 0
    
    #########################
    ## FUNDAMENTAL METHODS ##
    #########################
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
                self.last_error = 1
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
        header    = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
        message   = header + payload

        # print(f"Sending {len(message)} bytes: {message.hex()}")

        client = self.client
        if client is None:
            raise RuntimeError("Not connected to VAST API.")
        client.sendall(message)

        # Receive response header 
        hdr = client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr!r}")

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type  = struct.unpack("<I", hdr[12:16])[0]

        # Receive payload 
        payload_bytes = b""
        while len(payload_bytes) < total_len - 4:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload_bytes += chunk

        # print(f"Response msg_type={msg_type}, len={len(payload_bytes)}")
        return msg_type, payload_bytes

    def parse_payload(self, data: bytes) -> dict:
        """
        Parse the raw VAST payload (no header).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Payload must be bytes")

        result = {
            "msg_id":  None,
            "uint32":  [],
            "float64": [],
            "int32":   [],
            "strings": [],
            "uint64":  [],
        }

        p = 0
        while p < len(data):
            t = data[p]
            if t == 1:  # uint32
                val = struct.unpack_from("<I", data, p + 1)[0]
                result["uint32"].append(val)
                p += 5
            elif t == 2:  # double
                val = struct.unpack_from("<d", data, p + 1)[0]
                result["float64"].append(val)
                p += 9
            elif t == 3:  # zero-terminated string
                q = data.find(b"\x00", p + 1)
                if q == -1:
                    break
                result["strings"].append(data[p + 1:q].decode("utf-8", "ignore"))
                p = q + 1
            elif t == 4:  # int32
                val = struct.unpack_from("<i", data, p + 1)[0]
                result["int32"].append(val)
                p += 5
            elif t == 6:  # uint64
                val = struct.unpack_from("<Q", data, p + 1)[0]
                result["uint64"].append(val)
                p += 9
            else:
                # Unknown tag: stop to avoid misalignment
                break

        return result

    ############################
    #     GENERAL FUNCTIONS    #
    ############################

    def get_info(self) -> dict: 
        """ Get general information from VAST. """
        """ Returns a struct with the following fields if successful, or an empty struct [] if failed: """
        """
            info.datasizex X (horizontal) size of the data volume in voxels at full resolution
            info.datasizey Y (vertical) size of the data volume in voxels at full resolution
            info.datasizez Z (number of slices) size of the data volume in voxels
            info.voxelsizex X size of one voxel (in nm)
            info.voxelsizey Y size of one voxel (in nm)
            info.voxelsizez Z size of one voxel (in nm)
            info.cubesizex X size of the internal cubes used in VAST in voxels; always 16
            info.cubesizey Y size of the internal cubes used in VAST in voxels; always 16
            info.cubesizez Z size of the internal cubes used in VAST in voxels; always 16
            info.currentviewx Current view X coord in VAST in voxels at full res (window center)
            info.currentviewy Current view Y coord in VAST in voxels at full res (window center)
            info.currentviewz Current view Z coord in VAST in voxels (slice number)
            info.nrofmiplevels Number of mip levels of the current data set
        """
        msg_type, data = self.send_command(GETINFO)
        if msg_type == 21:  # error
            self.last_error = 21
            print("No segmentation or dataset loaded in VAST.")
            return {}
        try:
            parsed = self.parse_payload(data)
            
            u32 = parsed.get("uint32", [])
            f64 = parsed.get("float64", [])
            i32 = parsed.get("int32", [])

            if len(u32) < 7 or len(f64) < 3 or len(i32) < 3:
                print(f"Unexpected payload size: uint32={len(u32)}, float64={len(f64)}, int32={len(i32)}")
                return {}

            info = {
                "datasizex":     u32[0],
                "datasizey":     u32[1],
                "datasizez":     u32[2],
                "voxelsizex":    f64[0],
                "voxelsizey":    f64[1],
                "voxelsizez":    f64[2],
                "cubesizex":     u32[3],
                "cubesizey":     u32[4],
                "cubesizez":     u32[5],
                "currentviewx":  i32[0],
                "currentviewy":  i32[1],
                "currentviewz":  i32[2],
                "nrofmiplevels": u32[6],
            }

            return info
        
        except ValueError as e:
            self.last_error = 2
            print(f"Failed to parse info response: {data!r}, error: {e}")
        return {}
    
    def get_api_version(self) -> int:
        msg_type, data = self.send_command(GETAPIVERSION)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting API version: {errorCodes[msg_type]}")
            return 0
        try:
            version = struct.unpack("<I", data[1:5])[0]
            return version
        except struct.error as e:
            self.last_error = 2
            print(f"Failed to parse version response: {data!r}, error: {e}")
            return 0

    def get_hardware_info(self) -> dict:
        msg_type, data = self.send_command(GETHARDWAREINFO)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting hardware info: {errorCodes[msg_type]}")
            return {}
        try:
            parsed = self.parse_payload(data)

            u32    = parsed.get("uint32", [])
            f64    = parsed.get("float64", [])
            i32    = parsed.get("int32", [])
            texts  = parsed.get("strings", [])

            # Expected: 1 uint, 7 doubles, 0 ints, 5 text strings
            if len(u32) != 1 or len(f64) != 7 or len(i32) != 0 or len(texts) != 5:
                print(
                    f"Unexpected payload layout: "
                    f"uint32={len(u32)}, float64={len(f64)}, int32={len(i32)}, strings={len(texts)}"
                )
                return {}
            
            info = {
                "computername":                texts[0],
                "processorname":               texts[1],
                "processorspeed_ghz":          f64[0],
                "nrofprocessorcores":          u32[0],
                "tickspeedmhz":                f64[1],
                "mmxssecapabilities":          texts[2],
                "totalmemorygb":               f64[2],
                "freememorygb":                f64[3],
                "graphicscardname":            texts[3],
                "graphicsdedicatedvideomemgb": f64[4],
                "graphicsdedicatedsysmemgb":   f64[5],
                "graphicssharedsysmemgb":      f64[6],
                "graphicsrasterizerused":      texts[4],
            }

            return info
        except ValueError as e:
            self.last_error = 2
            print(f"Failed to parse hardware info response: {data!r}, error: {e}")
            return {}

    def get_last_error(self) -> int:
        """Query VAST for the most recent error code (like MATLAB getlasterror)."""
        return self.last_error

    ############################
    #      LAYER FUNCTION      #
    ############################

    def get_number_of_layers(self) -> int:
        """ Get the number of layers in the current VAST session. """
        msg_type, data = self.send_command(GETNROFLAYERS)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting number of layers: {errorCodes[msg_type]}")
            return 0
        parsed = self.parse_payload(data)
        u32 = parsed.get("uint32", [])

        if len(u32) == 1:
            self.last_error = 0
            return u32[0]
        else:
            self.last_error = 2
            print(f"Unexpected payload structure: uint32={len(u32)}")
            return 0

    def get_layer_info(self, layer_nr: int) -> dict:
        """
        Retrieve information about a specific layer in VAST.
        Corresponds to MATLAB getlayerinfo(layernr).
        """
        payload = struct.pack("<I", layer_nr)
        print(payload)
        msg_type, data = self.send_command(GETLAYERINFO, payload)
        if msg_type == 21:
            self.last_error = 21
            print("No segmentation or dataset loaded in VAST.")
            return {}
        parsed = self.parse_payload(data)
        print(parsed)
        return parsed

# import json
def main():
    vast = VASTControlClass(HOST,PORT)
    vast.connect()
    # print("api version: ", vast.get_api_version())
    # info = vast.get_info()
    # print(f"info is a {type(info[0])} as: {info[0]}")
    # hw_info = vast.get_hardware_info()
    # print(f"hw_info is a {type(hw_info[0])} as: {hw_info[0]}")
    n = vast.get_number_of_layers()
    print("Number of layers:", n)
    # for i in range(1,n+1):
    #     # print(i, vast.get_layer_info(i).get("name"), vast.get_layer_info(i).get("type"))
    #     print(i)
    #     vast.get_layer_info(i)
    # vast.get_layer_info(1)
    vast.get_layer_info(2)
    vast.get_layer_info(3)


    vast.disconnect()

if __name__ == "__main__":
    main()