import numpy as np
from skimage import measure
from scipy import ndimage
from math import floor
from VASTControlClass import VASTControlClass

def get_selected_seg_layer_name():
    """Get the name of the currently selected segment layer."""
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect")
        return None
    
    seg_layer = vast.get_selected_layer_nr()
    if seg_layer is None:
        print("No segment layer selected")
        vast.disconnect()
        return None
    
    
    vast.disconnect()
    return seg_layer

def get_child_tree_ids_seg(all_seg_data, parent_idx):
    """
    Recursively get all child segment indices.
    
    Args:
        all_seg_data: List of segment dictionaries
        parent_idx: Index of parent segment
    
    Returns:
        List of child indices
    """
    children = []
    parent_id = all_seg_data[parent_idx]['id']
    
    # Find all segments where this is the parent
    for i, seg in enumerate(all_seg_data):
        if seg['hierarchy'][0] == parent_id:  # hierarchy[0] is parent
            children.append(i)
            # Recursively add children of this child
            children.extend(get_child_tree_ids_seg(all_seg_data, i))
    
    return children

def extract_surfaces(params):
    """
    Extract 3D surfaces from VAST segmentation or screenshot data.
    
    Args:
        vast: Connected VASTControlClass instance
        params: Dictionary with export parameters
    """
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect")
        return None
    
    # Determine if extracting from segmentation or screenshots
    extract_seg = True
    if 5 <= params['extractwhich'] <= 10:
        extract_seg = False
    
    all_seg_data = None
    
    if extract_seg:
        # ===== SEGMENTATION EXPORT =====
        nr_segments = vast.get_number_of_segments()
        if nr_segments == 0:
            print("ERROR: Cannot export models from segments. No segmentation available.")
            return
        
        # Get segment data
        all_seg_data = vast.get_all_segment_data()
        if all_seg_data is None:
            print("Failed to get segment data")
            return
        names = vast.get_all_segment_names()
        
        # Get selected layer
        selected_layers = vast.get_selected_layer_nr()
        if not selected_layers or selected_layers.get('selected_segment_layer', -1) < 0:
            print("No segment layer selected")
            return
        
        seg_layer_nr = selected_layers['selected_segment_layer']
        if seg_layer_nr < 0:
            print("No segment layer selected")
            return
        selected_em_layer_nr = selected_layers.get('selected_em_layer', -1)

        # Get layer name
        seg_layer_info = vast.get_layer_info(seg_layer_nr)
        seg_layer_name = seg_layer_info['name'] if seg_layer_info else f"Layer_{seg_layer_nr}"
        
        # Remove background
        if names and len(names) > 0:
            names = names[1:]
        
        # Get max object number
        max_object_number = max([seg['id'] for seg in all_seg_data]) if all_seg_data else 0
        
        # Get data size at mip level
        mip_data_size = vast.get_data_size_at_mip(params['miplevel'], seg_layer_nr)
        
        
    else:
        names = []
        seg_layer_nr = -1
        selected_em_layer_nr = -1
        extract_which = params['extractwhich']
        
        if extract_which == 5:
            # RGB 50% isosurfaces
            names = ['Red Layer', 'Green Layer', 'Blue Layer']
            
        elif extract_which == 6:
            # Brightness 50%
            params['lev'] = [128]
            names = ['Brightness 128']
            
        elif extract_which == 7:
            # 16 brightness levels
            params['lev'] = list(range(8, 257, 16))  # 8, 24, 40, ..., 248
            names = [f'B{lev:03d}' for lev in params['lev']]
            
        elif extract_which == 8:
            # 32 brightness levels
            params['lev'] = list(range(4, 257, 8))  # 4, 12, 20, ..., 252
            names = [f'B{lev:03d}' for lev in params['lev']]
            
        elif extract_which == 9:
            # 64 brightness levels
            params['lev'] = list(range(2, 257, 4))  # 2, 6, 10, ..., 254
            names = [f'B{lev:03d}' for lev in params['lev']]
            
        elif extract_which == 10:
            # One object per color (up to 2^24 objects)
            # This will be determined later from actual screenshot data
            names = []
            # colorcounts would be allocated when processing screenshot data
    
    selected_layers = vast.get_selected_layer_nr()
    if not selected_layers:
        print("Failed to get selected layers")
        return
    
    selected_layer_nr = selected_layers['selected_layer']
    selected_em_layer_nr = selected_layers['selected_em_layer']
    selected_segment_layer_nr = selected_layers['selected_segment_layer']
    
    # Initialize Z scaling
    zscale = 1
    zmin = params['zmin']
    zmax = params['zmax']
    
    # Apply mip level scaling
    mip_scale_matrix = None
    if params['miplevel'] > 0:
        # Get mip scale factors from appropriate layer
        if extract_seg:
            mip_scale_matrix = vast.get_mipmap_scale_factors(selected_segment_layer_nr)
        else:
            mip_scale_matrix = vast.get_mipmap_scale_factors(selected_em_layer_nr)
        
        if mip_scale_matrix and params['miplevel'] <= len(mip_scale_matrix):
            # Get Z scale factor for this mip level
            zscale = mip_scale_matrix[params['miplevel'] - 1][2]  # 0-based indexing
        else:
            zscale = 1
        
        # Scale Z coordinates if needed
        if zscale != 1:
            zmin = int(zmin / zscale)
            zmax = int(zmax / zscale)
    xmin = params['xmin'] >> params['miplevel']  # Right bitshift
    xmax = (params['xmax'] >> params['miplevel']) - 1
    ymin = params['ymin'] >> params['miplevel']
    ymax = (params['ymax'] >> params['miplevel']) - 1
    
    # Get mip scale factors
    if params['miplevel'] > 0:
        if mip_scale_matrix and params['miplevel'] <= len(mip_scale_matrix):
            mipfactx = mip_scale_matrix[params['miplevel'] - 1][0]
            mipfacty = mip_scale_matrix[params['miplevel'] - 1][1]
            mipfactz = mip_scale_matrix[params['miplevel'] - 1][2]
        else:
            mipfactx = 1
            mipfacty = 1
            mipfactz = 1
    else:
        mipfactx = 1
        mipfacty = 1
        mipfactz = 1
    
    # Validation: check if volume is large enough
    if ((xmin == xmax) or (ymin == ymax) or (zmin == zmax)) and not params['closesurfaces']:
        print("ERROR: The surface extraction needs a volume which is at least two pixels "
              "wide in each direction. Please adjust region values, or enable 'closesurfaces'.")
        return
    
    if extract_seg:
        # Compute full name (including folder names) from name and hierarchy
        if params['includefoldernames'] and all_seg_data is not None:
            fullname = names.copy()
            for i in range(len(all_seg_data)):
                j = i
                # data(j,14) in MATLAB is hierarchy[0] (parent) in Python
                parent_id = all_seg_data[j]['hierarchy'][0]
                
                while parent_id != 0:  # Check if parent exists
                    # Find parent index (parent_id is 1-based segment ID)
                    parent_idx = next((idx for idx, seg in enumerate(all_seg_data) 
                                        if seg['id'] == parent_id), None)
                    
                    if parent_idx is None:
                        break
                    
                    j = parent_idx
                    fullname[i] = names[j] + '.' + fullname[i]
                    parent_id = all_seg_data[j]['hierarchy'][0]
            
            names = fullname
        
        # Compute list of objects to export
        extract_which = params['extractwhich']
        
        if extract_which == 1:
            # All segments individually, uncollapsed
            if all_seg_data:
                objects = [(seg['id'], seg['flags']) for seg in all_seg_data]
            else:
                objects = []
            vast.set_seg_translation([], [])
        
        elif extract_which == 2:
            # All segments, collapsed as in VAST
            # Get unique collapsed IDs
            if all_seg_data:
                collapsed_ids = list(set(seg['collapsednr'] for seg in all_seg_data))
                objects = [(cid, all_seg_data[cid]['flags']) for cid in collapsed_ids]
                
                # Set translation: map all segment IDs to their collapsed IDs
                source_array = [seg['id'] for seg in all_seg_data]
                target_array = [seg['collapsednr'] for seg in all_seg_data]
                vast.set_seg_translation(source_array, target_array)
            else:
                objects = []
        
        elif extract_which == 3:
            # Selected segment and children, uncollapsed
            # Find selected segments (flag bit 16 set = 65536)
            selected = [i for i, seg in enumerate(all_seg_data) 
                        if seg['flags'] & 65536] if all_seg_data else []
            
            if len(selected) == 0:
                # None selected: export all
                objects = [(seg['id'], seg['flags']) for seg in all_seg_data]
            else:
                # Add children
                selected_with_children = selected.copy()
                for sel_idx in selected:
                    children = get_child_tree_ids_seg(all_seg_data, sel_idx)
                    selected_with_children.extend(children)
                
                # Remove duplicates
                selected_with_children = list(set(selected_with_children))
                
                objects = [(all_seg_data[i]['id'], all_seg_data[i]['flags']) 
                            for i in selected_with_children]
                
                # Set translation to only show selected
                selected_ids = [all_seg_data[i]['id'] for i in selected_with_children]
                vast.set_seg_translation(selected_ids, selected_ids)
        
        elif extract_which == 4:
            # Selected segment and children, collapsed as in VAST
            # Find selected segments
            selected = [i for i, seg in enumerate(all_seg_data) 
                        if seg['flags'] & 65536] if all_seg_data else []
            
            if len(selected) == 0:
                # None selected: export all, collapsed
                selected_indices = list(range(len(all_seg_data)))
                collapsed_ids = list(set(seg['collapsednr'] for seg in all_seg_data))
            else:
                # Add children
                selected_with_children = selected.copy()
                for sel_idx in selected:
                    children = get_child_tree_ids_seg(all_seg_data, sel_idx)
                    selected_with_children.extend(children)
                
                selected_indices = list(set(selected_with_children))
                
                # Get unique collapsed IDs from selected
                collapsed_ids = list(set(all_seg_data[i]['collapsednr'] 
                                        for i in selected_indices))
            
            objects = [(cid, all_seg_data[cid]['flags']) for cid in collapsed_ids]
            
            # Set translation
            source_array = [all_seg_data[i]['id'] for i in selected_indices]
            target_array = [all_seg_data[i]['collapsednr'] for i in selected_indices]
            vast.set_seg_translation(source_array, target_array)
    
    nrxtiles = 0
    tilex1 = xmin
    while tilex1 <= xmax:
        tilex1 = tilex1 + params['blocksizex'] - params['overlap']
        nrxtiles += 1
    
    # Y tiles
    nrytiles = 0
    tiley1 = ymin
    while tiley1 <= ymax:
        tiley1 = tiley1 + params['blocksizey'] - params['overlap']
        nrytiles += 1
    
    # Z tiles
    nrztiles = 0
    tilez1 = zmin
    
    if params['slicestep'] == 1:
        # Use all slices
        slicenumbers = list(range(zmin, zmax + 1))  # +1 because range is exclusive
        
        while tilez1 <= zmax:
            tilez1 = tilez1 + params['blocksizez'] - params['overlap']
            nrztiles += 1
        
        blockslicenumbers = None  # Not needed when slicestep=1
    else:
        # Use every nth slice
        slicenumbers = list(range(zmin, zmax + 1, params['slicestep']))
        
        # Calculate number of Z blocks
        import math
        nrztiles = math.ceil(len(slicenumbers) / (params['blocksizez'] - params['overlap']))
        
        # Build block slice number list
        blockslicenumbers = []
        step = params['blocksizez'] - params['overlap']
        
        for p in range(0, len(slicenumbers), step):
            pe = min(p + params['blocksizez'], len(slicenumbers))
            blockslicenumbers.append(slicenumbers[p:pe])
    
    # Store in params for later use
    params['nrxtiles'] = nrxtiles
    params['nrytiles'] = nrytiles
    params['nrztiles'] = nrztiles
    params['slicenumbers'] = slicenumbers
    params['blockslicenumbers'] = blockslicenumbers
    
    print(f"Volume will be processed in {nrxtiles}x{nrytiles}x{nrztiles} = {nrxtiles*nrytiles*nrztiles} blocks")

    if params['usemipregionconstraint']:
        cxmin = xmin >> -params['mipregionmip']
        cxmax = xmax >> -params['mipregionmip']-1
        cymin = ymin >> -params['mipregionmip']
        cymax = ymax >> -params['mipregionmip']-1
        czmin = zmin
        czmax = zmax
        if params['mipregionpadding'] > 0:
            czscale = mip_scale_matrix[params['mipregionmip'] - 1][2] if mip_scale_matrix else 1
            czmin = floor(czmin/czscale)
            czmax = floor(czmax/czscale)
        
        if extract_seg:
            # Load complete region of segmentation source layer at constraint mip level (vdata.data.exportobj.mipregionmip equals param.mipregionmip)
            if params['slicestep'] == 1:
                mcsegimage, values, counts, bboxes = vast.get_seg_image_rle_decoded_bboxes(
                    params['mipregionmip'], cxmin, cxmax, cymin, cymax, czmin, czmax, False)
            else:
                s = list(range(czmin, czmax + 1, params['slicestep']))
                mcsegimage = np.zeros((cxmax - cxmin + 1, cymax - cymin + 1, len(s)), dtype=int)
                for i, z in enumerate(s):
                    mcsegslice, values, counts, bboxes = vast.get_seg_image_rle_decoded_bboxes(
                        params['mipregionmip'], cxmin, cxmax, cymin, cymax, z, z, False)
                    mcsegimage[:, :, z] = mcsegslice
        else:
            if params['slicestep'] == 1:
                mcsegimage = vast.get_screenshot_image(params['mipregionmip'], cxmin, cxmax, cymin, cymax, czmin, czmax, False)
                if mcsegimage is not None and mcsegimage.ndim == 4:
                    mcsegimage = np.transpose(np.sum(mcsegimage, axis=3) > 0, (1, 0, 2))
            else:
                s = list(range(czmin, czmax + 1, params['slicestep']))
                mcsegimage = np.zeros((cxmax - cxmin + 1, cymax - cymin + 1, len(s)), dtype=int)
                for i, z in enumerate(s):
                    mcsegslice = vast.get_screenshot_image(params['mipregionmip'], cxmin, cxmax, cymin, cymax, z, z, False)
                    if mcsegslice is not None and mcsegslice.ndim == 4:
                        mcsegimage[:, :, z] = np.transpose(np.sum(mcsegslice, axis=3) > 0, (1, 0))
                    else:
                        mcsegimage[:, :, i] = np.zeros((cxmax - cxmin + 1, cymax - cymin + 1), dtype=int)
        if mcsegimage is not None:
            mcsegimage = mcsegimage > 0

        # Dilate mask by region padding
        sz = int(params['mipregionpadding'] * 2 + 1)
        struct_element = np.ones((sz, sz, sz), dtype=bool)
        mcsegimage = ndimage.binary_dilation(mcsegimage, structure=struct_element)
        # Generate 3D matrix of block loading flags
        mc_loadflags = np.zeros((nrxtiles, nrytiles, nrztiles), dtype=bool)

        if mip_scale_matrix is not None:
            cmipfactx=mip_scale_matrix[params['mipregionmip']][0]/mip_scale_matrix[params['mipregionmip']][0];
            cmipfacty=mip_scale_matrix[params['mipregionmip']][1]/mip_scale_matrix[params['mipregionmip']][1];
            cmipfactz=mip_scale_matrix[params['mipregionmip']][2]/mip_scale_matrix[params['mipregionmip']][2];
        tilez1 = zmin

        for tz in range(nrztiles):
            tilez2 = tilez1 + params['blocksizez'] - 1
            if tilez2 > zmax:
                tilez2 = zmax
            tiley1 = ymin
            for ty in range(nrytiles):
                tiley2 = tiley1 + params['blocksizey'] - 1
                if tiley2 > ymax:
                    tiley2 = ymax
                tilex1 = xmin
                for tx in range(nrxtiles):
                    tilex2 = tilex1 + params['blocksizex'] - 1
                    if tilex2 > xmax:
                        tilex2 = xmax
                    
                    # Check if any voxel in this block is set in mcsegimage
                    cminx = max(0, int(np.floor((tilex1 - xmin) / cmipfactx)))
                    cmaxx = min(mcsegimage.shape[0], int(np.ceil((tilex2 - xmin) / cmipfactx)) + 1)
                    cminy = max(0, int(np.floor((tiley1 - ymin) / cmipfacty)))
                    cmaxy = min(mcsegimage.shape[1], int(np.ceil((tiley2 - ymin) / cmipfacty)) + 1)
                    cminz = max(0, int(np.floor((tilez1 - zmin) / cmipfactz)))
                    cmaxz = min(mcsegimage.shape[2], int(np.ceil((tilez2 - zmin) / cmipfactz)) + 1)
                    
                    cropregion = mcsegimage[cminx:cmaxx, cminy:cmaxy, cminz:cmaxz]
                    if cropregion.size > 0:
                        mc_loadflags[tx, ty, tz] = np.max(cropregion)
                    else:
                        mc_loadflags[tx, ty, tz] = 0
                    tilex1 += (params['blocksizex'] - params['overlap'])
                tiley1 += (params['blocksizey'] - params['overlap'])
            tilez1 += (params['blocksizez'] - params['overlap'])
        
    

# def export_segment_meshes(miplevel=0, output_file=None):
#     """Export segment as triangular mesh (OBJ/STL)."""
#     vast = VASTControlClass()
#     if not vast.connect():
#         print("Failed to connect")
#         return
    
#     info = vast.get_info()

#     # get number of segments
#     nr_segments = vast.get_number_of_segments()
#     if nr_segments == 0:  
#         print("ERROR: Cannot export models from segments. No segmentation available in VAST.")
#         return

#     extract_seg = True  

#     # Get all segment data
#     all_seg_data = vast.get_all_segment_data()  # Returns list of dicts
#     names = vast.get_all_segment_names()  # Returns list of strings

#     # Get selected segmentation layer
#     selected_layers = vast.get_selected_layer_nr()
#     if not selected_layers or selected_layers.get('selected_segment_layer', -1) < 0:
#         print("No segment layer selected")
#         vast.disconnect()
#         return

#     seg_layer_nr = selected_layers['selected_segment_layer']

#     # Get layer info to get the name
#     seg_layer_info = vast.get_layer_info(seg_layer_nr)
#     if seg_layer_info:
#         seg_layer_name = seg_layer_info['name']
#     else:
#         seg_layer_name = f"Layer_{seg_layer_nr}"

#     # Remove background name (first entry)
#     if names and len(names) > 0:
#         names = names[1:]  # Remove first element (Background)

#     # Get max object number
#     max_object_number = max([seg['id'] for seg in all_seg_data]) if all_seg_data else 0

#     # Get mip data size
#     mip_data_size = vast.get_data_size_at_mip(seg_layer_nr, miplevel)



#     # seg_data = vast.get_segment_data(segment_id)
#     # if not seg_data:
#     #     return None
    
#     # bbox = seg_data['boundingbox']
#     # minx, miny, minz, maxx, maxy, maxz = bbox
    
#     # print(f"Fetching segment {segment_id}...")
    
#     # voxels = vast.get_seg_image_rle_decoded(
#     #     miplevel=miplevel,
#     #     minx=minx, maxx=maxx,
#     #     miny=miny, maxy=maxy,
#     #     minz=minz, maxz=maxz
#     # )
    
#     # if voxels is None:
#     #     return None
    
#     # mask = (voxels == segment_id).astype(np.uint8)
    
#     # print("Running marching cubes...")
#     # verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    
#     # # Scale to nm
#     # scale = np.array([info['voxelsizex'], info['voxelsizey'], info['voxelsizez']])
#     # offset = np.array([minx, miny, minz])
    
#     # verts_scaled = verts * scale + offset * scale
    
#     # if output_file:
#     #     if output_file.endswith('.obj'):
#     #         save_obj(output_file, verts_scaled, faces)
#     #     elif output_file.endswith('.stl'):
#     #         save_stl(output_file, verts_scaled, faces)
#     #     print(f"Saved {output_file}")
    
#     # return verts_scaled, faces
#     vast.disconnect()

def export_skeleton(vast, anno_object_id, output_file=None):
    """
    Export skeleton as SWC or OBJ line mesh.
    
    Returns: (nodes, edges) where nodes has columns [id, x, y, z, radius, parent_id]
    """
    
    info = vast.get_info()
    if not info:
        return None
    
    # Select the annotation object
    if not vast.set_selected_anno_object_nr(anno_object_id):
        print(f"Failed to select annotation object {anno_object_id}")
        return None
    
    # Get node data
    node_data = vast.get_ao_node_data()
    if node_data is None:
        print("Failed to get node data")
        return None
    
    # Extract and scale coordinates
    scale = np.array([info['voxelsizex'], info['voxelsizey'], info['voxelsizez']])
    
    # Build node list: [id, x, y, z, radius, parent_id]
    nodes = []
    for i in range(len(node_data)):
        node_id = int(node_data[i, 0])
        x = node_data[i, 12] * scale[0]
        y = node_data[i, 13] * scale[1]
        # Z coordinate needs to be extracted separately if stored
        z = 0  # You may need to get Z from anchorpoint or object data
        radius = node_data[i, 11]
        parent_id = int(node_data[i, 5])  # parent index
        
        nodes.append([node_id, x, y, z, radius, parent_id])
    
    nodes = np.array(nodes)
    
    # Build edge list
    edges = []
    for i in range(len(nodes)):
        parent_idx = int(nodes[i, 5])
        if parent_idx >= 0 and parent_idx < len(nodes):
            edges.append([i, parent_idx])
    
    if output_file:
        if output_file.endswith('.swc'):
            save_swc(output_file, nodes)
        elif output_file.endswith('.obj'):
            save_skeleton_obj(output_file, nodes, edges)
        print(f"Saved {output_file}")
    
    return nodes, edges


def save_obj(filename, vertices, faces):
    """Save mesh as Wavefront OBJ."""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_skeleton_obj(filename, nodes, edges):
    """Save skeleton as OBJ line mesh."""
    with open(filename, 'w') as f:
        # Write vertices
        for node in nodes:
            f.write(f"v {node[1]} {node[2]} {node[3]}\n")
        
        # Write edges as lines
        for edge in edges:
            f.write(f"l {edge[0]+1} {edge[1]+1}\n")


def save_swc(filename, nodes):
    """Save skeleton in SWC format (standard neuron format)."""
    with open(filename, 'w') as f:
        f.write("# SWC format skeleton\n")
        f.write("# id type x y z radius parent\n")
        for node in nodes:
            node_id = int(node[0]) + 1  # SWC uses 1-based
            parent_id = int(node[5]) + 1 if node[5] >= 0 else -1
            f.write(f"{node_id} 0 {node[1]:.3f} {node[2]:.3f} {node[3]:.3f} {node[4]:.3f} {parent_id}\n")


def save_stl(filename, vertices, faces):
    """Save mesh as binary STL."""
    import struct
    
    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

def main():
    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect")
    
    # Get dataset info
    info = vast.get_info()
    if not info:
        print("No dataset loaded")
        vast.disconnect()

    vast.disconnect()

    params = {
        # Region to export
        'xmin': 0,
        'xmax': info['datasizex'] - 1,
        'ymin': 0,
        'ymax': info['datasizey'] - 1,
        'zmin': 0,
        'zmax': info['datasizez'] - 1,
        
        # Mip level and sampling
        'miplevel': 2,  # 0=full res, higher=lower res
        'slicestep': 1,  # Use every nth slice
        
        # Mip region constraint (optional)
        'usemipregionconstraint': False,
        'mipregionmip': info['nrofmiplevels'] - 1,
        'mipregionpadding': 1,
        
        # Processing block size
        'blocksizex': 1024,
        'blocksizey': 1024,
        'blocksizez': 64,
        'overlap': 1,
        
        # Scaling and units
        'xscale': 0.001,
        'yscale': 0.001,
        'zscale': 0.001,
        'xunit': info['voxelsizex'],  # nm
        'yunit': info['voxelsizey'],
        'zunit': info['voxelsizez'],
        
        # Output offset
        'outputoffsetx': 0,
        'outputoffsety': 0,
        'outputoffsetz': 0,
        
        # Options
        'invertz': True,
        'erodedilate': False,
        'closesurfaces': True,
        
        # Export mode
        'extractwhich': 1,  # 1=all segments uncollapsed, 2=collapsed, 
                           # 3=selected+children uncollapsed, 4=selected+children collapsed
                           # 5=RGB isosurfaces, 6=brightness isosurface, 
                           # 7/8/9=multi-level brightness, 10=one per color
        
        # File output
        'targetfileprefix': 'Segment_',
        'targetfolder': './vast_export',
        'fileformat': 1,  # 1=OBJ/MTL, 2=PLY
        'includefoldernames': True,
        'objectcolors': 1,  # 1=VAST colors, 2=volume-based colormap
        
        # Advanced
        'skipmodelgeneration': False,
        'disablenetwarnings': True,
        'write3dsmaxloader': False,
        'savesurfacestats': False,
        'surfacestatsfile': 'surfacestats.txt',
    }

    # Export segment mesh
    extract_surfaces(params)
    
    # Export skeleton
    # export_skeleton(vast, anno_object_id=1, output_file="skeleton_1.swc")






if __name__ == "__main__":
    main()
    