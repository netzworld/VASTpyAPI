import socket
import struct
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

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
    
    @staticmethod
    def _encode_uint32(value: int) -> bytes:
        """
        Equivalent of MATLAB bytesfromuint32(uint32(value)).
        Format (from parse.m tags):
        [1][4 bytes of uint32 little-endian]
        """
        return b"\x01" + struct.pack("<I", value & 0xFFFFFFFF)

    @staticmethod
    def _encode_int32(value: int) -> bytes:
        """bytesfromint32(int32(value)) | for other APIs later."""
        return b"\x04" + struct.pack("<i", value)
        
    @staticmethod
    def _encode_double(value: float) -> bytes:
        # tag 2 + float64 LE
        return b"\x02" + struct.pack("<d", float(value))
    
    def send_command(self, msg_id: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """
        Send a binary command to the VAST API and return the message type and response bytes.

        Each message follows the format:
        [0..3]   = b"VAST"
        [4..11]  = uint64 length of data following (payload + 4 bytes for msg_id)
        [12..15] = uint32 message ID (little-endian)
        [16..]   = payload (optional)
        """
        client = self.client
        if client is None:
            raise RuntimeError("Not connected to VAST API.")
        
        total_len = len(payload) + 4
        header    = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
        message   = header + payload

        # print(f"Sending {len(message)} bytes: {message.hex()}")

        
        client.sendall(message)

        # Receive response header 
        hdr = client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr!r}")

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type  = struct.unpack("<I", hdr[12:16])[0]

        expected = total_len - 4
        payload_bytes = bytearray()
        while len(payload_bytes) < expected:
            chunk = client.recv(expected - len(payload_bytes))
            if not chunk:
                raise RuntimeError("Incomplete payload from VAST.")
            payload_bytes.extend(chunk)

        # print(f"Response msg_type={msg_type}, len={len(payload_bytes)}")
        return msg_type, bytes(payload_bytes)

    def parse_payload(self, indata: bytes) -> Dict[str, Any]:
        """
        Python equivalent of the inner part of MATLAB parse(obj, indata).

        indata is assumed to be ONLY the typed payload (no 'VAST' header), returned by self.send_command().

        Returns a dict:
            {
                "ints":    [int32, ...],
                "uints":   [uint32, ...],
                "doubles": [float64, ...],
                "text":    [str, ...],      # all 0-terminated text chunks
                "last_text": str | "",      # last inchardata
                "uint64s": [int, ...],      # uint64 values
            }
        """
        ints: List[int]      = []
        uints: List[int]     = []
        doubles: List[float] = []
        texts: List[str]     = []
        last_text: str       = ""
        uint64s: List[int]   = []

        p = 0
        n = len(indata)
        while p < n:
            t   = indata[p]

            if t   == 1:  # int32
                if p + 5 > n:
                    break
                val = struct.unpack("<I", indata[p+1:p+5])[0]
                uints.append(val)
                p  += 5
            elif t == 2: # double | float64
                if p + 9 > n:
                    break
                val = struct.unpack("<d", indata[p+1:p+9])[0]
                doubles.append(val)
                p  += 9
            elif t == 3: # 0-terminated text
                q = p + 1
                while q < n and indata[q] != 0:
                    q += 1
                if q  >= n or indata[q]   != 0:
                    break # abort parsing
                text_bytes = indata[p+1:q]
                try:
                    s = text_bytes.decode("utf-8", errors="replace")
                except Exception:
                    s = "".join(chr(b) for b in text_bytes)
                last_text = s
                texts.append(s)
                p = q + 1 # skip null terminator
            elif t == 4: # int32
                if p + 5 > n:
                    break
                val = struct.unpack("<i", indata[p+1:p+5])[0]
                ints.append(val)
                p  += 5
            elif t == 5: # uint64
                if p + 9 > n:
                    break
                val = struct.unpack("<Q", indata[p+1:p+9])[0]
                uint64s.append(val)
                p  += 9
            else:
                break # unknown type
        return {
            "ints":    ints,
            "uints":   uints,
            "doubles":  doubles,
            "text":  texts,
            "last_text": last_text,
            "uint64s":   uint64s,
        }
    
    ############################
    #     GENERAL FUNCTIONS    #
    ############################

    def get_info(self) -> dict: 
        """
        Get general information from VAST.
        
        Returns dict with dataset info, or empty dict on failure.
        Fields: datasizex/y/z, voxelsizex/y/z, cubesizex/y/z, 
                currentviewx/y/z, nrofmiplevels
        """
        msg_type, data = self.send_command(GETINFO)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        doubles = parsed.get("doubles", [])
        ints = parsed.get("ints", [])
        
        if len(uints) == 7 and len(doubles) == 3 and len(ints) == 3:
            self.last_error = 0
            return {
                "datasizex":     uints[0],
                "datasizey":     uints[1],
                "datasizez":     uints[2],
                "voxelsizex":    doubles[0],
                "voxelsizey":    doubles[1],
                "voxelsizez":    doubles[2],
                "cubesizex":     uints[3],
                "cubesizey":     uints[4],
                "cubesizez":     uints[5],
                "currentviewx":  ints[0],
                "currentviewy":  ints[1],
                "currentviewz":  ints[2],
                "nrofmiplevels": uints[6],
            }
        else:
            self.last_error = 2
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
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])

        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            print(f"Unexpected payload structure: uint32={len(uints)}")
            return 0

    def get_layer_info(self, layer_nr: int) -> dict:
        """
        Retrieve information about a specific layer in VAST.
        Corresponds to MATLAB getlayerinfo(layernr).
        """
        payload = self._encode_uint32(layer_nr)
        msg_type, data = self.send_command(GETLAYERINFO, payload)

        parsed  = self.parse_payload(data)
        ints    = parsed.get("ints", [])
        uints   = parsed.get("uints", [])
        doubles = parsed.get("doubles", [])
        name    = parsed.get("last_text", "")

        if not (len(ints) == 8 and len(uints) == 7 and len(doubles) == 3 and name):
            self.last_error = 2
            print(f"Unexpected payload structure: ints={len(ints)}, uints={len(uints)}, doubles={len(doubles)}, text={len(name)}")
            return {}

        self.last_error = 0
        
        layer_type_map = {
            0: "Image",
            1: "Segmentation",
            2: "Annotation",
            3: "VSVR image",
            6: "VSVI image",
            7: "Tool",
        }
        layer_type = layer_type_map.get(ints[0], "Other")
        layerinfo = {
            "type":             layer_type,
            "editable":         ints[1],
            "visible":          ints[2],
            "brightness":       ints[3],
            "contrast":         ints[4],

            "opacitylevel":     doubles[0],
            "brightnesslevel":  doubles[1],
            "contrastlevel":    doubles[2],

            "blendmode":        ints[5],
            "blendoradd":       ints[6],

            "tintcolor":        uints[0],
            "name":             name,

            "redtargetcolor":   uints[1],
            "greentargetcolor": uints[2],
            "bluetargetcolor":  uints[3],
            "bytesperpixel":    uints[4],

            "ischanged":        ints[7],
            "inverted":         uints[5],
            "solomode":         uints[6],
        }

        return layerinfo        

    def set_layer_info(self, layer_nr: int, layer_info: dict) -> bool:
        """
        Set information for a specific layer in VAST.
        Corresponds to MATLAB setlayerinfo(layernr, layerinfo).
        """
        if not layer_info:
            self.last_error = 0
            return True
        xflags = 0
        msg = self._encode_uint32(layer_nr)

        def has(key: str) -> bool:
            return key in layer_info and layer_info[key] is not None
        
        # 1  editable (int32)
        if has("editable"):
            msg += self._encode_int32(layer_info["editable"])
            xflags |= 1

        # 2  visible (int32)
        if has("visible"):
            msg += self._encode_int32(layer_info["visible"])
            xflags |= 2

        # 4  brightness (int32)
        if has("brightness"):
            msg += self._encode_int32(layer_info["brightness"])
            xflags |= 4

        # 8  contrast (int32)
        if has("contrast"):
            msg += self._encode_int32(layer_info["contrast"])
            xflags |= 8

        # 16 opacitylevel (double)
        if has("opacitylevel"):
            msg += self._encode_double(layer_info["opacitylevel"])
            xflags |= 16

        # 32 brightnesslevel (double)
        if has("brightnesslevel"):
            msg += self._encode_double(layer_info["brightnesslevel"])
            xflags |= 32

        # 64 contrastlevel (double)
        if has("contrastlevel"):
            msg += self._encode_double(layer_info["contrastlevel"])
            xflags |= 64

        # 128 blendmode (int32)
        if has("blendmode"):
            msg += self._encode_int32(layer_info["blendmode"])
            xflags |= 128

        # 256 blendoradd (int32)
        if has("blendoradd"):
            msg += self._encode_int32(layer_info["blendoradd"])
            xflags |= 256

        # 512 tintcolor (uint32)
        if has("tintcolor"):
            msg += self._encode_uint32(layer_info["tintcolor"])
            xflags |= 512

        # 1024 redtargetcolor (uint32)
        if has("redtargetcolor"):
            msg += self._encode_uint32(layer_info["redtargetcolor"])
            xflags |= 1024

        # 2048 greentargetcolor (uint32)
        if has("greentargetcolor"):
            msg += self._encode_uint32(layer_info["greentargetcolor"])
            xflags |= 2048

        # 4096 bluetargetcolor (uint32)
        if has("bluetargetcolor"):
            msg += self._encode_uint32(layer_info["bluetargetcolor"])
            xflags |= 4096

        # 8192 inverted (uint32)
        if has("inverted"):
            msg += self._encode_uint32(layer_info["inverted"])
            xflags |= 8192

        # 16384 solomode (uint32)
        if has("solomode"):
            msg += self._encode_uint32(layer_info["solomode"])
            xflags |= 16384

        # If after all that xflags is still 0, nothing is set → no-op
        if xflags == 0:
            self.last_error = 0
            return True

        # Final payload: [bytesfromuint32(xflags), msg...]
        payload = self._encode_uint32(xflags) + msg

        # Send command and look at error code in header
        msg_type, data = self.send_command(SETLAYERINFO, payload)

        success = (msg_type == 1)
        if success:
            self.last_error = 0
        else:
            # Optional: we could parse 'data' to find an error code or
            # add a get_last_error() call here. For now, just flag nonzero.
            self.last_error = -1

        return success

    def get_mipmap_scale_factors(self, layer_nr: int) -> List[List[int]]:
        """
        Get the mipmap scale factors for a specific layer in VAST.
        Corresponds to MATLAB getmipmapscalefactors(layernr).
        """
        payload = self._encode_uint32(layer_nr)

        msg_type, raw = self.send_command(GETMIPMAPSCALEFACTORS, payload)

        if len(raw) < 4 or (len(raw) % 4) != 0:
            # At least one uint32 needed, and payload must be multiple of 4 bytes
            self.last_error = 2  # "unexpected data received"
            return []

        # Interpret payload as an array of uint32 (little-endian)
        count = len(raw) // 4
        uid = struct.unpack("<{}I".format(count), raw)

        num_levels = uid[0]  # same as uid(1) in MATLAB

        # We expect 1 + 3*num_levels entries total
        if 1 + 3 * num_levels > len(uid):
            self.last_error = 2
            return []

        # Build matrix including mip level 0
        matrix: List[List[int]] = []
        idx = 1  # start from uid(2) in MATLAB (0-based index 1)

        for _ in range(num_levels):
            sx = uid[idx]
            sy = uid[idx + 1]
            sz = uid[idx + 2]
            matrix.append([sx, sy, sz])
            idx += 3

        # Drop first row: mip level 0 (always [1,1,1])
        if matrix:
            matrix = matrix[1:]

        self.last_error = 0
        return matrix

    def set_selected_layer_nr(self, layer_nr: int) -> bool:
        """
        Set the selected layer in VAST.
        Corresponds to MATLAB setselectedlayernr(layernr).
        """
        payload = self._encode_uint32(layer_nr)
        msg_type, data = self.send_command(SETSELECTEDLAYERNR, payload)
        
        success = (msg_type == 1)
        if success:
            self.last_error = 0
        else:
            self.last_error = -1

        return success
    
    def get_selected_layer_nr(self) -> dict:
        """
        Get the selected layer number in VAST.
        Corresponds to MATLAB getselectedlayernr().
        """
        msg_type, data = self.send_command(GETSELECTEDLAYERNR)

        # msg_type is 'res' (1 = success, 0 = failure), not an error code
        if msg_type != 1:
            self.last_error = -1  
            return {}

        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])

        # MATLAB cases:
        #   if nrinints == 3:
        #       [sel, sel_em, sel_seg] = inintdata(1..3)
        #   elseif nrinints == 5:
        #       sel     = inintdata(1)
        #       sel_em  = inintdata(2)
        #       sel_seg = inintdata(4)
        if len(ints) == 3:
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[2]
        elif len(ints) == 5:
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[3]
        else:
            # Mirror MATLAB: unexpected data received
            self.last_error = 2  # "Unexpected data received from VAST - API mismatch?"
            return {}

        # If we got here, treat as success; -1 is a *valid* "none" value.
        self.last_error = 0
        return {
            "selected_layer": selected_layer,
            "selected_em_layer": selected_em_layer,
            "selected_segment_layer": selected_segment_layer,
        }
    
    def get_selected_layer_nr2(self) -> dict:
        msg_type, data = self.send_command(GETSELECTEDLAYERNR)

        # msg_type is 'res': 1 = success, 0 = failure
        if msg_type != 1:
            self.last_error = -1  # generic failure; can refine with getlasterror later
            return {}

        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])

        # Defaults as in MATLAB
        selected_layer = -1
        selected_em_layer = -1
        selected_anno_layer = -1
        selected_segment_layer = -1
        selected_tool_layer = -1

        if len(ints) == 3:
            # Backwards compatibility case
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[2]
            # anno/tool remain -1
        elif len(ints) == 5:
            selected_layer        = ints[0]
            selected_em_layer     = ints[1]
            selected_anno_layer   = ints[2]
            selected_segment_layer= ints[3]
            selected_tool_layer   = ints[4]
        else:
            # Unexpected layout
            self.last_error = 2  # "unexpected data received"
            return {}

        self.last_error = 0
        return {
            "selected_layer":         selected_layer,
            "selected_em_layer":      selected_em_layer,
            "selected_anno_layer":    selected_anno_layer,
            "selected_segment_layer": selected_segment_layer,
            "selected_tool_layer":    selected_tool_layer,
        }

    ############################
    #   SEGMENTATION FUNCTIONS #
    ############################

    def get_number_of_segments(self) -> int:
        """Get the number of segments in the current segmentation."""
        msg_type, data = self.send_command(GETNUMBEROFSEGMENTS)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def get_segment_data(self, segment_id: int) -> dict:
        """
        Get metadata for a specific segment.
        
        Returns dict with:
            id, flags, col1, col2, anchorpoint, hierarchy, collapsednr, boundingbox
        """
        payload = self._encode_uint32(segment_id)
        msg_type, data = self.send_command(GETSEGMENTDATA, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        uints = parsed.get("uints", [])
        
        if len(ints) == 9 and len(uints) == 9:
            self.last_error = 0
            return {
                "id": uints[0],
                "flags": uints[1],
                "col1": uints[2],
                "col2": uints[3],
                "anchorpoint": ints[0:3],
                "hierarchy": list(uints[4:8]),
                "collapsednr": uints[8],
                "boundingbox": ints[3:9],
            }
        else:
            self.last_error = 2
            return {}

    def get_all_segment_data(self) -> List[dict]:
        """
        Get metadata for all segments.
        
        Returns list of dicts, each containing segment data.
        """
        msg_type, data = self.send_command(GETALLSEGMENTDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        # For large data, might need chunked reading
        if len(data) < 4:
            self.last_error = 2
            return []
        
        # Parse as uint32 array
        arr = np.frombuffer(data, dtype=np.uint32)
        iarr = np.frombuffer(data, dtype=np.int32)
        
        num_segments = arr[0]
        segments = []
        
        sp = 1  # start position (0-based in Python)
        for i in range(num_segments):
            seg = {
                "id": i,
                "flags": arr[sp],
                "col1": arr[sp + 1],
                "col2": arr[sp + 2],
                "anchorpoint": [int(iarr[sp + 3]), int(iarr[sp + 4]), int(iarr[sp + 5])],
                "hierarchy": [int(arr[sp + 6]), int(arr[sp + 7]), int(arr[sp + 8]), int(arr[sp + 9])],
                "collapsednr": arr[sp + 10],
                "boundingbox": [int(iarr[sp + 11]), int(iarr[sp + 12]), int(iarr[sp + 13]),
                               int(iarr[sp + 14]), int(iarr[sp + 15]), int(iarr[sp + 16])],
            }
            segments.append(seg)
            sp += 17
        
        self.last_error = 0
        return segments

    def get_seg_image_raw(self, miplevel: int, minx: int, maxx: int, 
                          miny: int, maxy: int, minz: int, maxz: int,
                          immediate: bool = False, request_load: bool = False) -> Optional[bytes]:
        """
        Get raw segmentation image data as uint16 array.
        
        Args:
            miplevel: Mip level to retrieve
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            immediate: Use immediate mode (don't wait for disk cache)
            request_load: Request loading from disk if not in cache
            
        Returns:
            Raw bytes (uint16 little-endian) or None on error
        """
        cmd = GETSEGIMAGERAWIMMEDIATE if immediate else GETSEGIMAGERAW
        
        if immediate:
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz) + self._encode_uint32(int(request_load))
        else:
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(cmd, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_seg_image_rle(self, miplevel: int, minx: int, maxx: int,
                          miny: int, maxy: int, minz: int, maxz: int,
                          surf_only: bool = False, immediate: bool = False,
                          request_load: bool = False) -> Optional[bytes]:
        """
        Get RLE-encoded segmentation image data.
        
        Returns:
            RLE-encoded bytes (uint16 pairs: value, count) or None on error
        """
        if immediate:
            cmd = GETSEGIMAGERLEIMMEDIATE
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz) + self._encode_uint32(int(request_load))
        else:
            cmd = GETSEGIMAGESURFRLE if surf_only else GETSEGIMAGERLE
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(cmd, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_seg_image_rle_decoded(self, miplevel: int, minx: int, maxx: int,
                                   miny: int, maxy: int, minz: int, maxz: int,
                                   surf_only: bool = False, immediate: bool = False,
                                   request_load: bool = False):
        """
        Get segmentation image with RLE decoding applied.
        
        Returns:
            numpy array of shape (maxy-miny+1, maxx-minx+1, maxz-minz+1) dtype uint16
        """
        import numpy as np
        
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        
        if rle_data is None:
            return None
        
        # Decode RLE
        rle = np.frombuffer(rle_data, dtype=np.uint16)
        
        # Calculate expected size
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        expected_size = size_x * size_y * size_z
        
        result = np.zeros(expected_size, dtype=np.uint16)
        dp = 0
        
        for sp in range(0, len(rle), 2):
            val = rle[sp]
            count = rle[sp + 1]
            result[dp:dp + count] = val
            dp += count
        
        # Reshape to (X, Y, Z) then transpose to (Y, X, Z) to match MATLAB
        result = result.reshape(size_x, size_y, size_z)
        result = np.transpose(result, (1, 0, 2))
        
        return result

    def set_seg_translation(self, source_array: List[int], target_array: List[int]) -> bool:
        """
        Set segmentation translation table for image retrieval.
        
        Maps source segment IDs to target IDs during image transfer.
        Pass empty arrays to clear translation.
        """
        if len(source_array) != len(target_array):
            self.last_error = 50
            return False
        
        # Interleave source and target arrays
        translate = []
        for src, tgt in zip(source_array, target_array):
            translate.append(src)
            translate.append(tgt)
        
        payload = b""
        for val in translate:
            payload += self._encode_uint32(val)
        
        msg_type, data = self.send_command(SETSEGTRANSLATION, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success
    
    ############################
    #   ANNOTATION FUNCTIONS   #
    ############################

    def get_anno_layer_nr_of_objects(self) -> tuple:
        """Get number of annotation objects and first object number."""
        msg_type, data = self.send_command(GETANNOLAYERNROFOBJECTS)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0, -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 2:
            self.last_error = 0
            return uints[0], uints[1]
        else:
            self.last_error = 2
            return 0, -1

    def get_ao_node_data(self) -> Optional[np.ndarray]:
        """
        Get all skeleton node data from selected annotation layer.
        
        Returns numpy array with columns:
        0: node index (DFS number)
        1: isselected flag
        2: edge flags
        3: has label flag
        4: reserved
        5-10: parent, child1, child2, prev, next, nrofchildren (uint32, -1 means none)
        11: radius (double)
        12-13: x, y coordinates (uint32)
        """
        msg_type, data = self.send_command(GETAONODEDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        import numpy as np
        
        # Parse as uint32 array
        arr = np.frombuffer(data, dtype=np.uint32)
        
        if len(arr) < 1:
            self.last_error = 2
            return None
        
        num_nodes = arr[0]
        node_data = np.zeros((num_nodes, 14))
        
        sp = 1
        for i in range(num_nodes):
            node_data[i, 0] = i  # DFS number
            node_data[i, 1] = arr[sp] & 0xFF  # isselected
            node_data[i, 2] = (arr[sp] >> 8) & 0xFF  # edgeflags
            node_data[i, 3] = (arr[sp] >> 16) & 0xFF  # haslabel
            node_data[i, 4] = (arr[sp] >> 24) & 0xFF  # reserved
            
            # Hierarchy (convert 0xFFFFFFFF to -1)
            for j in range(6):
                val = arr[sp + 1 + j]
                node_data[i, 5 + j] = -1 if val == 0xFFFFFFFF else val
            
            # Radius (double at position sp+7, sp+8)
            radius_bytes = data[4 * (sp + 7):4 * (sp + 9)]
            node_data[i, 11] = np.frombuffer(radius_bytes, dtype=np.float64)[0]
            
            # X, Y coordinates
            node_data[i, 12] = arr[sp + 9]
            node_data[i, 13] = arr[sp + 10]
            
            sp += 11
        
        self.last_error = 0
        return node_data

    def get_anno_layer_object_data(self) -> List[dict]:
        """Get metadata for all annotation objects."""
        msg_type, data = self.send_command(GETANNOLAYEROBJECTDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint32)
        
        if len(arr) < 1:
            return []
        
        num_objects = arr[0]
        objects = []
        
        sp = 1
        for i in range(num_objects):
            obj = {
                "id": i + 1,
                "type": arr[sp] & 0xFFFF,  # 0=folder, 1=skeleton
                "flags": (arr[sp] >> 16) & 0xFFFF,
                "col1": arr[sp + 1],
                "col2": arr[sp + 2],
                "anchorpoint": [arr[sp + 3], arr[sp + 4], arr[sp + 5]],
                "hierarchy": [arr[sp + 6], arr[sp + 7], arr[sp + 8], arr[sp + 9]],
                "collapsednr": arr[sp + 10],
                "boundingbox": [arr[sp + 11], arr[sp + 12], arr[sp + 13], 
                            arr[sp + 14], arr[sp + 15], arr[sp + 16]],
            }
            objects.append(obj)
            sp += 21
        
        self.last_error = 0
        return objects

    def set_selected_anno_object_nr(self, object_id: int) -> bool:
        """Select an annotation object by ID."""
        payload = self._encode_uint32(object_id)
        msg_type, data = self.send_command(SETSELECTEDANNOOBJECTNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success