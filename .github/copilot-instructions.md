# VAST Python API - AI Coding Agent Guide

## Project Overview
This is a Python interface to the **VAST (Volume Annotation and Segmentation Tool)** API, a neuroscience image analysis tool for segmenting and annotating volumetric EM (electron microscopy) data. The project bridges Python applications with VAST's TCP/IP-based API.

## Architecture

### Core Component: VASTControlClass
[VASTControlClass.py](VASTControlClass.py) is the main abstraction layer with **3,593 lines** implementing:
- **TCP/IP Communication**: Socket-based binary protocol with VAST at `127.0.0.1:22081`
- **Message Protocol**: Custom binary format (`VAST` header + length + msg_id + payload)
- **Type Encoding/Decoding**: 10 data types (uint32, int32, uint64, double, string, matrices, string arrays)
- **80+ API Commands**: Image retrieval (raw/RLE), segment manipulation, layer management, annotation objects

### Key API Command Categories (constants defined at top):
- **Image Retrieval**: `GETSEGIMAGERAW`, `GETSEGIMAGERLE`, `GETEMIMAGERAW`
- **Segment Operations**: `GETNUMBEROFSEGMENTS`, `GETSEGMENTNAME`, `SETSEGMENTNAME`, `ADDSEGMENT`
- **Layer Management**: `GETNROFLAYERS`, `ADDNEWLAYER`, `LOADLAYER`, `SAVELAYER`
- **Annotation Objects**: Hierarchical tree nodes (skeletons) for tracing neural structures
- **Visualization**: View coordinates/zoom, drawing properties, screenshots

### Data Serialization Protocol
The binary protocol uses type tags (0-9) with structured encoding:
- Type 0-4: Scalars (uint32, int32, uint64, double, string)
- Type 5-8: Matrices (xsize, ysize, data)
- Type 9: String arrays (nrofstrings, totaltextlength, concatenated strings)

## Usage Patterns

### Connection Lifecycle
```python
vast = VASTControlClass()  # Defaults to 127.0.0.1:22081
if not vast.connect():
    print("Failed to connect")
    # VAST application must be running and accessible
    
try:
    # Use API methods
    pass
finally:
    vast.disconnect()
```

### Data Retrieval Pattern
```python
# Most image retrieval methods accept miplevel and bbox (minx, maxx, miny, maxy, minz, maxz)
raw_data = vast.get_seg_image_raw(miplevel=2, minx=0, maxx=99, miny=0, maxy=99, minz=0, maxz=0)
seg_array = np.frombuffer(raw_data, dtype=np.uint16)

# RLE format returns alternating (value, count) pairs
rle_data = vast.get_seg_image_rle(...)
rle_array = np.frombuffer(rle_data, dtype=np.uint16)
```

### Error Handling
Always check `last_error` after API calls (error codes defined in `errorCodes` dict):
```python
if not result:
    error_code = vast.get_last_error()
    print(errorCodes.get(error_code, "Unknown error"))
```

## Key Files
- **[main.py](main.py)**: Example connection and basic API usage
- **[test.py](test.py)**: Test suite for image retrieval (raw/RLE format validation)
- **[exportSkeleton.py](exportSkeleton.py)**: Skeleton/annotation export with mesh merging
- **VAST_package_1_5_0/**: Reference Matlab implementation (vasttools.m, VASTControlClass.m)

## Important Constraints
- **Single Host Assumption**: All calls use same host/port defined at initialization
- **Blocking Socket**: `connect()` has timeout parameter; all I/O is synchronous
- **RLE Format**: If compression makes data larger, API returns error; fallback to raw format
- **Layer Context**: Many operations depend on `selected_layer_nr`; verify before operations

## Common Workflows
1. **Image Analysis**: Load layer → get raw/RLE image → convert to numpy array → process
2. **Segmentation**: Get segment data → modify → set via API → refresh view
3. **Annotation Export**: Get annotation objects → build tree structure → export as mesh/skeleton
