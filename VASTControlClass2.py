import socket
import struct
import numpy as np

HOST = "127.0.0.1"
PORT = 22081

class VASTControlClass:
    """
    Python implementation of VASTControlClass.
    Communicates with VAST (Volume Annotation and Segmentation Tool) via TCP/IP.
    """

    def __init__(self):
        self.sock = None
        self.isconnected = False
        self.lasterror = 0
        self.timeout = 1.0

        # Parsing buffers (mimicking MATLAB structure)
        self.inres = 0
        self.nrinints = 0
        self.inintdata = []
        self.nrinuints = 0
        self.inuintdata = []
        self.nrindoubles = 0
        self.indoubledata = []
        self.nrinchars = 0
        self.inchardata = ""
        self.nrintext = 0
        self.intextdata = []
        self.nrinuint64s = 0
        self.inuint64data = []
        
        self.indata = b"" # Raw byte buffer

        # --- CONSTANTS ---
        self.GETINFO = 1
        self.GETNUMBEROFSEGMENTS = 2
        self.GETSEGMENTDATA = 3
        self.GETSEGMENTNAME = 4
        self.SETANCHORPOINT = 5
        self.SETSEGMENTNAME = 6
        self.SETSEGMENTCOLOR = 7
        self.GETVIEWCOORDINATES = 8
        self.GETVIEWZOOM = 9
        self.SETVIEWCOORDINATES = 10
        self.SETVIEWZOOM = 11
        self.GETNROFLAYERS = 12
        self.GETLAYERINFO = 13
        self.GETALLSEGMENTDATA = 14
        self.GETALLSEGMENTNAMES = 15
        self.SETSELECTEDSEGMENTNR = 16
        self.GETSELECTEDSEGMENTNR = 17
        self.SETSELECTEDLAYERNR = 18
        self.GETSELECTEDLAYERNR = 19
        self.GETSEGIMAGERAW = 20
        self.GETSEGIMAGERLE = 21
        self.GETSEGIMAGESURFRLE = 22
        self.SETSEGTRANSLATION = 23
        self.GETSEGIMAGERAWIMMEDIATE = 24
        self.GETSEGIMAGERLEIMMEDIATE = 25
        self.GETEMIMAGERAW = 30
        self.GETEMIMAGERAWIMMEDIATE = 31
        self.REFRESHLAYERREGION = 32
        self.GETPIXELVALUE = 33
        self.GETSCREENSHOTIMAGERAW = 40
        self.GETSCREENSHOTIMAGERLE = 41
        self.SETSEGIMAGERAW = 50
        self.SETSEGIMAGERLE = 51
        self.SETSEGMENTBBOX = 60
        self.GETFIRSTSEGMENTNR = 61
        self.GETHARDWAREINFO = 62
        self.ADDSEGMENT = 63
        self.MOVESEGMENT = 64
        self.GETDRAWINGPROPERTIES = 65
        self.SETDRAWINGPROPERTIES = 66
        self.GETFILLINGPROPERTIES = 67
        self.SETFILLINGPROPERTIES = 68
        
        self.GETANNOLAYERNROFOBJECTS = 70
        self.GETANNOLAYEROBJECTDATA = 71
        self.GETANNOLAYEROBJECTNAMES = 72
        self.ADDNEWANNOOBJECT = 73
        self.MOVEANNOOBJECT = 74
        self.REMOVEANNOOBJECT = 75
        self.SETSELECTEDANNOOBJECTNR = 76
        self.GETSELECTEDANNOOBJECTNR = 77
        self.GETAONODEDATA = 78
        self.GETAONODELABELS = 79
        
        self.SETSELECTEDAONODEBYDFSNR = 80
        self.SETSELECTEDAONODEBYCOORDS = 81
        self.GETSELECTEDAONODENR = 82
        self.ADDAONODE = 83
        self.MOVESELECTEDAONODE = 84
        self.REMOVESELECTEDAONODE = 85
        self.SWAPSELECTEDAONODECHILDREN = 86
        self.MAKESELECTEDAONODEROOT = 87
        self.SPLITSELECTEDSKELETON = 88
        self.WELDSKELETONS = 89
        
        self.GETANNOOBJECT = 90
        self.SETANNOOBJECT = 91
        self.ADDANNOOBJECT = 92
        self.GETCLOSESTAONODEBYCOORDS = 93
        self.GETAONODEPARAMS = 94
        self.SETAONODEPARAMS = 95
        
        self.GETAPIVERSION = 100
        self.GETAPILAYERSENABLED = 101
        self.SETAPILAYERSENABLED = 102
        self.GETSELECTEDAPILAYERNR = 103
        self.SETSELECTEDAPILAYERNR = 104
        
        self.GETCURRENTUISTATE = 110
        self.GETERRORPOPUPSENABLED = 112
        self.SETERRORPOPUPSENABLED = 113
        self.SETUIMODE = 114
        self.SHOWWINDOW = 115
        self.SET2DVIEWORIENTATION = 116
        self.GETPOPUPSENABLED = 117
        self.SETPOPUPSENABLED = 118
        
        self.ADDNEWLAYER = 120
        self.LOADLAYER = 121
        self.SAVELAYER = 122
        self.REMOVELAYER = 123
        self.MOVELAYER = 124
        self.SETLAYERINFO = 125
        self.GETMIPMAPSCALEFACTORS = 126
        
        self.EXECUTEFILL = 131
        self.EXECUTELIMITEDFILL = 132
        self.EXECUTECANVASPAINTSTROKE = 133
        self.EXECUTESTARTAUTOSKELETONIZATION = 134
        self.EXECUTESTOPAUTOSKELETONIZATION = 135
        self.EXECUTEISAUTOSKELETONIZATIONDONE = 136
        
        self.SETTOOLPARAMETERS = 151

    # --------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # --------------------------------------------------------------------------

    def connect(self, host, port, timeout=1000):
        """Connect to VAST server."""
        try:
            self.sock = socket.create_connection((host, port), timeout=timeout/1000.0)
            self.sock.settimeout(timeout/1000.0)
            self.timeout = timeout/1000.0
            self.isconnected = True
            return 1
        except Exception as e:
            self.lasterror = 1 # Connection error
            self.isconnected = False
            return 0

    def disconnect(self):
        """Disconnect from VAST server."""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None
        self.isconnected = False
        return 1

    def getlasterror(self):
        return self.lasterror

    # --------------------------------------------------------------------------
    # CORE API FUNCTIONS
    # --------------------------------------------------------------------------

    def getinfo(self):
        """Reads out general information from VAST[cite: 77]."""
        self._send_message(self.GETINFO, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        
        info = {}
        if res == 1:
            if self.nrinuints >= 7 and self.nrindoubles >= 3 and self.nrinints >= 3:
                info['datasizex'] = self.inuintdata[0]
                info['datasizey'] = self.inuintdata[1]
                info['datasizez'] = self.inuintdata[2]
                info['voxelsizex'] = self.indoubledata[0]
                info['voxelsizey'] = self.indoubledata[1]
                info['voxelsizez'] = self.indoubledata[2]
                info['cubesizex'] = self.inuintdata[3]
                info['cubesizey'] = self.inuintdata[4]
                info['cubesizez'] = self.inuintdata[5]
                info['currentviewx'] = self.inintdata[0]
                info['currentviewy'] = self.inintdata[1]
                info['currentviewz'] = self.inintdata[2]
                info['nrofmiplevels'] = self.inuintdata[6]
            else:
                self.lasterror = 2 # Unexpected data
                return None, 0
        return info, res

    def getsegmentdata(self, segment_id):
        """Gets data for a specific segment ID[cite: 3]."""
        self._send_message(self.GETSEGMENTDATA, self._bytes_from_uint32(segment_id))
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        
        data = {}
        if res == 1:
            if self.nrinints >= 9 and self.nrinuints >= 9:
                data['id'] = self.inuintdata[0]
                data['flags'] = self.inuintdata[1]
                data['col1'] = self.inuintdata[2] # Primary color
                data['col2'] = self.inuintdata[3] # Secondary color
                data['anchorpoint'] = self.inintdata[0:3]
                data['hierarchy'] = self.inuintdata[4:8] # parent, child, prev, next
                data['collapsednr'] = self.inuintdata[8]
                data['boundingbox'] = self.inintdata[3:9]
            else:
                self.lasterror = 2
                return None, 0
        return data, res

    def setsegmentcolor8(self, segment_id, r1, g1, b1, p1, r2, g2, b2, p2):
        """Sets segment color using 8-bit RGBA components[cite: 7]."""
        v1 = (int(p1) & 255) + ((int(b1) & 255) << 8) + ((int(g1) & 255) << 16) + ((int(r1) & 255) << 24)
        v2 = (int(p2) & 255) + ((int(b2) & 255) << 8) + ((int(g2) & 255) << 16) + ((int(r2) & 255) << 24)
        
        payload = self._bytes_from_uint32([segment_id, v1, v2])
        self._send_message(self.SETSEGMENTCOLOR, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def getsegimageraw(self, miplevel, minx, maxx, miny, maxy, minz, maxz):
        """Gets raw segmentation image data[cite: 20]."""
        # Pack request parameters
        params = [miplevel, minx, maxx, miny, maxy, minz, maxz]
        self._send_message(self.GETSEGIMAGERAW, self._bytes_from_uint32(params))
        
        # Read large data block
        self._read_data_block()
        
        # Manually parse raw image data (skipping standard parse for speed on large buffers)
        # Header size is handled in _read_data_block, result is in self.indata
        # However, standard parsing logic splits header and result code
        self._parse_header(self.indata)
        res = self._process_error()

        segimage = None
        if res == 1:
            # Data starts at offset 16 (12 header + 4 res code)
            raw_data = self.indata[16:]
            # Convert raw bytes to uint16 numpy array
            segimage = np.frombuffer(raw_data, dtype=np.uint16)
            
            # Reshape logic matches MATLAB: [width, height, depth]
            # MATLAB: reshape([maxx-minx+1, maxy-miny+1, maxz-minz+1])
            # Note: Python uses C-order by default, MATLAB uses Fortran-order.
            # To match MATLAB's linear indexing, we might need to reshape carefully.
            dims = (int(maxz-minz+1), int(maxy-miny+1), int(maxx-minx+1)) 
            # Note: VAST sends data X, then Y, then Z. 
            # Numpy reshape with 'F' (Fortran) order mimics MATLAB memory layout
            segimage = segimage.reshape((int(maxx-minx+1), int(maxy-miny+1), int(maxz-minz+1)), order='F')
            
        return segimage, res

    def getemimageraw(self, layernr, miplevel, minx, maxx, miny, maxy, minz, maxz):
        """Gets raw EM (grayscale) image data[cite: 30]."""
        params = [layernr, miplevel, minx, maxx, miny, maxy, minz, maxz]
        self._send_message(self.GETEMIMAGERAW, self._bytes_from_uint32(params))
        self._read_data_block()
        self._parse_header(self.indata)
        res = self._process_error()
        
        emimage = None
        if res == 1:
            raw_data = self.indata[16:]
            emimage = np.frombuffer(raw_data, dtype=np.uint8)
            # Reshape assuming 1 byte per pixel for basic EM
            # Actual shape depends on layer properties (1 vs 3 vs 4 bytes), 
            # but usually it's [X, Y, Z].
            try:
                 emimage = emimage.reshape((int(maxx-minx+1), int(maxy-miny+1), int(maxz-minz+1)), order='F')
            except ValueError:
                pass # Return flat if reshape fails (e.g. multi-byte pixel formats)
        
        return emimage, res

    def getlayerinfo(self, layernr):
        """Returns information about the layer with the given number[cite: 13]."""
        self._send_message(self.GETLAYERINFO, self._bytes_from_uint32(layernr))
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        
        layerinfo = {}
        if res == 1:
            if self.nrinints >= 8 and self.nrinuints >= 7 and self.nrindoubles >= 3:
                layerinfo['type'] = self.inintdata[0]
                layerinfo['editable'] = self.inintdata[1]
                layerinfo['visible'] = self.inintdata[2]
                layerinfo['brightness'] = self.inintdata[3]
                layerinfo['contrast'] = self.inintdata[4]
                layerinfo['opacitylevel'] = self.indoubledata[0]
                layerinfo['brightnesslevel'] = self.indoubledata[1]
                layerinfo['contrastlevel'] = self.indoubledata[2]
                layerinfo['blendmode'] = self.inintdata[5]
                layerinfo['blendoradd'] = self.inintdata[6]
                layerinfo['tintcolor'] = self.inuintdata[0]
                layerinfo['name'] = self.inchardata
                layerinfo['redtargetcolor'] = self.inuintdata[1]
                layerinfo['greentargetcolor'] = self.inuintdata[2]
                layerinfo['bluetargetcolor'] = self.inuintdata[3]
                layerinfo['bytesperpixel'] = self.inuintdata[4]
                layerinfo['ischanged'] = self.inintdata[7]
                layerinfo['inverted'] = self.inuintdata[5]
                layerinfo['solomode'] = self.inuintdata[6]
            else:
                self.lasterror = 2
                return None, 0
        return layerinfo, res
    
    def getannolayerobjectdata(self):
        """Returns info about all annotation objects[cite: 71]."""
        self._send_message(self.GETANNOLAYEROBJECTDATA, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        
        objects = []
        if res == 1:
            # Parsing the flattened array from inuintdata
            # MATLAB: converts raw bytes to uint32/int32. 
            # Our _parse method already populates inuintdata.
            # However, GETANNOLAYEROBJECTDATA returns a generic "Tag 5" (matrix) or series of ints?
            # Looking at MATLAB code: it re-parses raw bytes specifically for this function.
            # It expects Tag 5 (matrix) or just a stream of uint32s. 
            # The MATLAB code explicitly uses `typecast` on `indata(17:end)`.
            
            raw_payload = self.indata[16:]
            if len(raw_payload) == 0:
                return [], res

            # VAST returns raw uint32s here, not tagged parameters
            # First uint32 is count
            count = struct.unpack('<I', raw_payload[0:4])[0]
            
            # The rest is the data matrix. 
            # Format: 29 uint32s/int32s per object. 
            # We read them as uint32 for simplicity and cast to int where needed (like bbox)
            
            # Create numpy view
            full_data = np.frombuffer(raw_payload, dtype=np.uint32)
            # full_data[0] is count. Data starts at full_data[1]
            
            if len(full_data) < 1 + count * 29:
                self.lasterror = 2
                return [], 0

            # Reshape: [Count, 29]
            # Note: MATLAB code skips first element (count), then iterates.
            data_matrix = full_data[1:1+count*29].reshape((count, 29))
            
            for row in data_matrix:
                obj = {}
                obj['id'] = row[0] # Actually loop index in MATLAB, but row[0] usually maps to ID
                obj['type'] = row[1] & 0xFFFF
                obj['flags'] = (row[1] >> 16) & 0xFFFF
                # Color 1
                obj['col1_r'] = (row[2] >> 24) & 0xFF
                obj['col1_g'] = (row[2] >> 16) & 0xFF
                obj['col1_b'] = (row[2] >> 8) & 0xFF
                obj['col1_p'] = row[2] & 0xFF
                # ... mapping continues as per Table A.5
                obj['anchor'] = row[4:7].view(np.int32) # XYZ
                obj['hierarchy'] = row[7:11] # parent, child, prev, next
                obj['bbox'] = row[12:18].view(np.int32)
                objects.append(obj)

        return objects, res

    def getapiversion(self):
        self._send_message(self.GETAPIVERSION, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 2:
            return (self.inuintdata[0], self.inuintdata[1]), res
        return None, res

    def getviewcoordinates(self):
        self._send_message(self.GETVIEWCOORDINATES, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinints >= 3:
            return self.inintdata[0:3], res
        return None, res

    def setviewcoordinates(self, x, y, z):
        payload = self._bytes_from_int32([x, y, z])
        self._send_message(self.SETVIEWCOORDINATES, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def getviewzoom(self):
        self._send_message(self.GETVIEWZOOM, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrindoubles >= 1:
            return self.indoubledata[0], res
        return None, res

    def setviewzoom(self, zoom):
        payload = self._bytes_from_double(zoom)
        self._send_message(self.SETVIEWZOOM, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def getcurrentuistate(self):
        self._send_message(self.GETCURRENTUISTATE, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1:
            state = {
                'selectedlayernr': self.inuintdata[0] if self.nrinuints > 0 else 0,
                'selectedsegmentnr': self.inuintdata[1] if self.nrinuints > 1 else 0,
                'uimode': self.inintdata[0] if self.nrinints > 0 else 0
            }
            return state, res
        return None, res
    
    # --------------------------------------------------------------------------
    # HELPER FUNCTIONS (INTERNAL)
    # --------------------------------------------------------------------------

    def _send_message(self, message_nr, payload):
        """
        Constructs and sends the VAST binary message.
        Format: 'VAST' + Length(64bit) + MsgNr(32bit) + Payload
        """
        if isinstance(payload, list):
            # Flatten list of bytes if necessary
            payload = b"".join(payload)
        
        length = len(payload) + 4 # Payload + MsgNr size
        
        # Header: 'VAST'
        header = b'VAST'
        # Length: 64-bit unsigned, Little Endian
        len_bytes = struct.pack('<Q', length)
        # MsgNr: 32-bit unsigned, Little Endian
        msg_nr_bytes = struct.pack('<I', message_nr)
        
        full_message = header + len_bytes + msg_nr_bytes + payload
        
        try:
            if self.sock is None:
                self.lasterror = 1
                self.isconnected = False
                return 0
            self.sock.sendall(full_message)
            return 1
        except:
            self.lasterror = 1
            self.isconnected = False
            return 0

    def _read_data_block(self):
        """Reads the response from VAST."""
        self.indata = b""
        if not self.sock:
            return

        try:
            # 1. Read Header (12 bytes: 'VAST' + 8 bytes Length)
            header_buf = self._recv_n(12)
            if not header_buf: return

            if header_buf[0:4] != b'VAST':
                self.lasterror = 2
                return

            data_len = struct.unpack('<Q', header_buf[4:12])[0]
            
            # 2. Read the rest of the data (Message ID + Payload)
            # data_len includes the 4 bytes for the result code/msg ID equivalent
            payload_buf = self._recv_n(data_len)
            if not payload_buf: 
                print("payload_buf is None")
                return
            self.indata = header_buf + payload_buf
            
        except Exception as e:
            self.lasterror = 1
            print(f"Socket Error: {e}")

    def _recv_n(self, n):
        """Helper to recv exactly n bytes."""
        data = b''
        if self.sock is None:
            return None
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _parse_header(self, data):
        """Parses just the header and result code."""
        self.parseheaderok = 0
        if len(data) < 16: return
        
        # Check VAST tag
        if data[0:4] != b'VAST': return
        
        # Result code is at bytes 12-16 (int32)
        self.inres = struct.unpack('<i', data[12:16])[0]
        self.parseheaderok = 1

    def _parse(self, data):
        """Parses the tagged binary data stream."""
        # Reset buffers
        self.inres = 0
        self.nrinints = 0; self.inintdata = []
        self.nrinuints = 0; self.inuintdata = []
        self.nrindoubles = 0; self.indoubledata = []
        self.nrintext = 0; self.intextdata = []
        self.inchardata = ""
        self.nrinuint64s = 0; self.inuint64data = []

        if len(data) < 16: return
        
        self._parse_header(data)
        
        # Start parsing parameters at byte 16
        p = 16
        limit = len(data)
        
        while p < limit:
            tag = data[p]
            
            if tag == 1: # uint32
                if p + 5 > limit: break
                val = struct.unpack('<I', data[p+1:p+5])[0]
                self.inuintdata.append(val)
                self.nrinuints += 1
                p += 5
                
            elif tag == 2: # double
                if p + 9 > limit: break
                val = struct.unpack('<d', data[p+1:p+9])[0]
                self.indoubledata.append(val)
                self.nrindoubles += 1
                p += 9
                
            elif tag == 3: # null-terminated string
                q = p + 1
                while q < limit and data[q] != 0:
                    q += 1
                
                # Decode string
                s_bytes = data[p+1:q]
                try:
                    s_val = s_bytes.decode('utf-8', errors='replace')
                except:
                    s_val = str(s_bytes)
                    
                self.inchardata = s_val # Store last string here mimicking MATLAB
                self.intextdata.append(s_val)
                self.nrintext += 1
                p = q + 1
                
            elif tag == 4: # int32
                if p + 5 > limit: break
                val = struct.unpack('<i', data[p+1:p+5])[0]
                self.inintdata.append(val)
                self.nrinints += 1
                p += 5
            
            elif tag == 6: # uint64
                if p + 9 > limit: break
                val = struct.unpack('<Q', data[p+1:p+9])[0]
                self.inuint64data.append(val)
                self.nrinuint64s += 1
                p += 9
                
            else:
                # Unknown tag or end of stream
                break

    def _process_error(self):
        """Sets lasterror based on inres and returned data."""
        self.lasterror = 0
        if self.inres == 0:
            if self.nrinuints >= 1:
                self.lasterror = self.inuintdata[0]
            else:
                self.lasterror = 1 # Unknown error
        return self.inres

    # --------------------------------------------------------------------------
    # DATA PACKING HELPERS
    # --------------------------------------------------------------------------

    def _bytes_from_uint32(self, value):
        """Tag 1: Packs int/list of ints as VAST uint32."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            value = [value]
        res = b""
        for v in value:
            res += struct.pack('<B', 1) + struct.pack('<I', int(v))
        return res

    def _bytes_from_int32(self, value):
        """Tag 4: Packs int/list of ints as VAST int32."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            value = [value]
        res = b""
        for v in value:
            res += struct.pack('<B', 4) + struct.pack('<i', int(v))
        return res

    def _bytes_from_double(self, value):
        """Tag 2: Packs float/list of floats as VAST double."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            value = [value]
        res = b""
        for v in value:
            res += struct.pack('<B', 2) + struct.pack('<d', float(v))
        return res

    def _bytes_from_text(self, value):
        """Tag 3: Packs string as VAST text."""
        if isinstance(value, str):
            value = value.encode('utf-8')
        return struct.pack('<B', 3) + value + b'\x00'

    def _bytes_from_data(self, value):
        """Tag 5: Packs binary blob."""
        # Value should be bytes or numpy array
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        length = len(value)
        # Tag 5 + Length(uint32) + Data
        return struct.pack('<B', 5) + struct.pack('<I', length) + value
    

    # --------------------------------------------------------------------------
    # LAYER FUNCTIONS
    # --------------------------------------------------------------------------

    def getnroflayers(self):
        self._send_message(self.GETNROFLAYERS, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 1:
            return self.inuintdata[0], res
        return 0, res

    def getselectedlayernr(self):
        self._send_message(self.GETSELECTEDLAYERNR, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 1:
            return self.inuintdata[0], res
        return 0, res

    def setselectedlayernr(self, layernr):
        self._send_message(self.SETSELECTEDLAYERNR, self._bytes_from_uint32(layernr))
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def addnewlayer(self, layer_type, name):
        """Types: 0=EM/Image, 1=Segmentation"""
        payload = self._bytes_from_uint32(layer_type) + self._bytes_from_text(name)
        self._send_message(self.ADDNEWLAYER, payload)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return (self.inuintdata[0] if self.nrinuints > 0 else -1), res

    def removelayer(self, layernr):
        self._send_message(self.REMOVELAYER, self._bytes_from_uint32(layernr))
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def movelayer(self, from_idx, to_idx):
        payload = self._bytes_from_uint32([from_idx, to_idx])
        self._send_message(self.MOVELAYER, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()
    
    # --------------------------------------------------------------------------
    # SEGMENTATION FUNCTIONS
    # --------------------------------------------------------------------------
    def getnumberofsegments(self):
        self._send_message(self.GETNUMBEROFSEGMENTS, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 1:
            return self.inuintdata[0], res
        return 0, res

    def getsegmentname(self, segment_id):
        self._send_message(self.GETSEGMENTNAME, self._bytes_from_uint32(segment_id))
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return self.inchardata, res

    def setsegmentname(self, segment_id, name):
        payload = self._bytes_from_uint32(segment_id) + self._bytes_from_text(name)
        self._send_message(self.SETSEGMENTNAME, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def getallsegmentnames(self):
        self._send_message(self.GETALLSEGMENTNAMES, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return self.intextdata, res

    def getselectedsegmentnr(self):
        self._send_message(self.GETSELECTEDSEGMENTNR, b"")
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 1:
            return self.inuintdata[0], res
        return 0, res

    def setselectedsegmentnr(self, segment_id):
        self._send_message(self.SETSELECTEDSEGMENTNR, self._bytes_from_uint32(segment_id))
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def addsegment(self, parent_id, prev_sibling_id, name):
        payload = self._bytes_from_uint32([parent_id, prev_sibling_id]) + self._bytes_from_text(name)
        self._send_message(self.ADDSEGMENT, payload)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return (self.inuintdata[0] if self.nrinuints > 0 else 0), res
    
    # --------------------------------------------------------------------------
    # VOXEL DATA TRANSFER FUNCTIONS
    # --------------------------------------------------------------------------

    def setsegimageraw(self, miplevel, minx, maxx, miny, maxy, minz, maxz, data_array):
        """
        Sends raw uint16 segmentation data to VAST.
        data_array should be a numpy array of uint16.
        """
        params = [miplevel, minx, maxx, miny, maxy, minz, maxz]
        payload = self._bytes_from_uint32(params)
        
        # Tag 5 for binary blob
        # Ensure data is in Fortran order (X, Y, Z) to match VAST
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array, dtype=np.uint16)
        
        raw_bytes = data_array.astype(np.uint16).tobytes(order='F')
        payload += self._bytes_from_data(raw_bytes)
        
        self._send_message(self.SETSEGIMAGERAW, payload)
        self._read_data_block()
        self._parse(self.indata)
        return self._process_error()

    def getpixelvalue(self, layernr, miplevel, x, y, z):
        """Returns the value of a single voxel."""
        params = [layernr, miplevel, x, y, z]
        self._send_message(self.GETPIXELVALUE, self._bytes_from_uint32(params))
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        if res == 1 and self.nrinuints >= 1:
            return self.inuintdata[0], res
        return 0, res

    def getsegimagerle(self, miplevel, minx, maxx, miny, maxy, minz, maxz):
        """Requests RLE compressed segmentation data."""
        params = [miplevel, minx, maxx, miny, maxy, minz, maxz]
        self._send_message(self.GETSEGIMAGERLE, self._bytes_from_uint32(params))
        self._read_data_block()
        self._parse_header(self.indata)
        res = self._process_error()
        
        if res == 1:
            # Data starts after header (12) + result code (4)
            # The payload contains RLE blocks. Parsing RLE into a full 3D array 
            # in Python is best done via a helper or specific logic depending on your needs.
            return self.indata[16:], res
        return None, res