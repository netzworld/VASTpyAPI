import numpy as np
from skimage import measure
from scipy import ndimage
from math import floor
import re
from typing import Tuple, List
import time
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


def merge_meshes(f1, v1, f2, v2):
    """
    Merge two meshes defined by (f1, v1) and (f2, v2).
    
    Merges meshes by identifying and merging duplicate vertices in the overlapping region,
    then re-indexing face indices appropriately.
    
    Args:
        f1: Face indices array for mesh 1 (Nx3)
        v1: Vertex coordinates array for mesh 1 (Mx3)
        f2: Face indices array for mesh 2 (Nx3)
        v2: Vertex coordinates array for mesh 2 (Mx3)
    
    Returns:
        tuple: (merged_faces, merged_vertices) where faces are re-indexed
    """
    # Handle empty meshes
    if v1.size == 0:
        return f2.copy(), v2.copy()
    
    if v2.size == 0:
        return f1.copy(), v1.copy()
    
    # Convert to numpy arrays if needed
    f1 = np.asarray(f1, dtype=np.int32)
    v1 = np.asarray(v1, dtype=np.float32)
    f2 = np.asarray(f2, dtype=np.int32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    nrofvertices1 = v1.shape[0]
    nrofvertices2 = v2.shape[0]
    
    # Adjust f2 indices by number of vertices in v1
    f2 = f2 + nrofvertices1
    
    # Find overlapping vertex region
    minv1 = np.min(v1, axis=0)
    maxv1 = np.max(v1, axis=0)
    minv2 = np.min(v2, axis=0)
    maxv2 = np.max(v2, axis=0)
    
    ovmin = np.maximum(minv1, minv2)
    ovmax = np.minimum(maxv1, maxv2)
    
    # Find vertices in overlap zone for v1
    mask1 = ((v1[:, 0] >= ovmin[0]) & (v1[:, 0] <= ovmax[0]) &
             (v1[:, 1] >= ovmin[1]) & (v1[:, 1] <= ovmax[1]) &
             (v1[:, 2] >= ovmin[2]) & (v1[:, 2] <= ovmax[2]))
    ov1_indices = np.where(mask1)[0]
    
    # Find vertices in overlap zone for v2
    mask2 = ((v2[:, 0] >= ovmin[0]) & (v2[:, 0] <= ovmax[0]) &
             (v2[:, 1] >= ovmin[1]) & (v2[:, 1] <= ovmax[1]) &
             (v2[:, 2] >= ovmin[2]) & (v2[:, 2] <= ovmax[2]))
    ov2_indices = np.where(mask2)[0]
    
    # If no overlap, concatenate meshes
    if ov2_indices.size == 0:
        f = np.vstack([f1, f2])
        v = np.vstack([v1, v2])
        return f, v
    
    # Find matching vertices in overlapping regions
    deletevertex = np.zeros(nrofvertices2, dtype=bool)
    
    # Use faster loopless version with intersect equivalent
    # Find common vertices in overlap zones
    ov1_verts = v1[ov1_indices]
    ov2_verts = v2[ov2_indices]
    
    # Find intersecting vertices (common coordinates)
    common_rows = []
    i1a_list = []
    i2a_list = []
    
    for i, v_ov1_idx in enumerate(ov1_indices):
        v_ov1 = v1[v_ov1_idx]
        # Check if this vertex exists in ov2
        for j, v_ov2_idx in enumerate(ov2_indices):
            v_ov2 = v2[v_ov2_idx]
            if np.allclose(v_ov1, v_ov2, atol=1e-6):
                i1a_list.append(v_ov1_idx)
                i2a_list.append(v_ov2_idx)
                common_rows.append(v_ov1)
                break
    
    # Link duplicate vertices
    if len(i2a_list) > 0:
        i1a = np.array(i1a_list, dtype=np.int32)
        i2a = np.array(i2a_list, dtype=np.int32)
        
        # Remap f2 to use v1 vertices for common vertices
        aov2 = i2a
        kov2 = aov2 + nrofvertices1
        
        # For each face in f2, replace vertices that match
        for idx in range(len(aov2)):
            v1_idx = i1a[idx]
            v2_idx = kov2[idx]
            # Replace all occurrences of v2_idx in f2 with v1_idx
            f2[f2 == v2_idx] = v1_idx
            deletevertex[aov2[idx]] = True
    
    # Re-index faces in f2 to account for deleted vertices
    z = np.arange(nrofvertices1, dtype=np.int32)
    z = np.append(z, np.zeros(nrofvertices2, dtype=np.int32))
    
    zp = nrofvertices1
    for sp in range(nrofvertices2):
        if not deletevertex[sp]:
            z[nrofvertices1 + sp] = zp
            zp += 1
    
    f2d = z[f2]
    
    # Ensure f2d is 2D array
    if f2d.ndim == 1:
        f2d = f2d.reshape(1, -1)
    
    # Delete unused vertices from v2 and concatenate
    v2d = v2[~deletevertex]
    
    f = np.vstack([f1, f2d])
    v = np.vstack([v1, v2d])
    
    return f, v


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
    max_object_number = 0
    
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
    
    objects = None
    
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
                    cmaxy = min(mcsegimage.shape[1] if mcsegimage.ndim > 1 else 1, int(np.ceil((tiley2 - ymin) / cmipfacty)) + 1)
                    cminz = max(0, int(np.floor((tilez1 - zmin) / cmipfactz)))
                    cmaxz = min(mcsegimage.shape[2] if mcsegimage.ndim > 2 else 1, int(np.ceil((tilez2 - zmin) / cmipfactz)) + 1)
                    
                    cropregion = mcsegimage[cminx:cmaxx, cminy:cmaxy, cminz:cmaxz]
                    if cropregion.size > 0:
                        mc_loadflags[tx, ty, tz] = np.max(cropregion)
                    else:
                        mc_loadflags[tx, ty, tz] = 0
                    tilex1 += (params['blocksizex'] - params['overlap'])
                tiley1 += (params['blocksizey'] - params['overlap'])
            tilez1 += (params['blocksizez'] - params['overlap'])
        
    if extract_seg:
        shape = (max_object_number, params['nrxtiles'], params['nrytiles'], params['nrztiles'])
        params['farray'] = np.empty(shape, dtype=object)
        params['varray'] = np.empty(shape, dtype=object)
        if objects is not None:
            params['objects'] = np.array(objects)
            params['object_volume'] = np.zeros((params['objects'].shape[0], 1)) 
    else:
        if(params['extractwhich'] == 10):
            # One object per color (up to 2^24 = 16,777,216 objects)
            num_colors = 256 * 256 * 256
            # Use sparse matrix for fvindex (color -> vertex/face indices mapping)
            from scipy.sparse import lil_matrix
            params['fvindex'] = lil_matrix((num_colors, (params['nrxtiles'] + 1) * (params['nrytiles'] + 1) * (params['nrztiles'] + 1)), dtype=int)
            
            # Object volume array: 64 MB for 256^3 objects
            params['object_volume'] = np.zeros((num_colors, 1))
            params['objects'] = None  # Will be determined from screenshot colors
        
        else:
            # Brightness isosurfaces (extractwhich 5-9)
            num_objects = len(names)
            shape = (num_objects, params['nrxtiles'], params['nrytiles'], params['nrztiles'])
            params['farray'] = np.empty(shape, dtype=object)
            params['varray'] = np.empty(shape, dtype=object)
            
            # Objects: [(object_id, flags), ...]
            # For brightness: IDs are 1 to num_objects
            params['objects'] = np.column_stack((
                np.arange(1, num_objects + 1),
                np.zeros(num_objects, dtype=int)
            ))
            
            params['object_volume'] = np.zeros((num_objects, 1))

    tilez1 = zmin
    tz = 0
    block_nr = 0
    while tz <= nrztiles:
        tilez2 = tilez1 + params['blocksizez'] - 1
        if tilez2 > zmax:
            tilez2 = zmax
        tilezs = tilez2 - tilez1 + 1
        tiley = ymin
        ty = 0
        while ty <= nrytiles:
            tiley2 = tiley1 + params['blocksizey'] - 1
            if tiley2 > ymax:
                tiley2 = ymax
            tileys = tiley2 - tiley1 + 1
            tilex = xmin
            tx = 0
            while tx <= nrxtiles:
                tilex2 = tilex1 + params['blocksizex'] - 1
                if tilex2 > xmax:
                    tilex2 = xmax
                tilexs = tilex2 - tilex1 + 1

                if extract_seg:
                    if not params['usemipregionconstraint'] or mc_loadflags[tx, ty, tz]:
                        print(f'Exporting Surfaces ...',('Loading Segmentation Cube (%d,%d,%d) of (%d,%d,%d)...',tx,ty,tz,nrxtiles,nrytiles,nrztiles))
                        # Call function to process this block
                        if params['erodedilate']:
                            edx = np.ones((3,2), dtype=int)
                            if tilex1 == 0:
                                edx[0,0] = 0
                            if tilex2 >= mip_data_size[0]:
                                edx[0,1] = 0
                            if tiley1 == 0:
                                edx[1,0] = 0
                            if tiley2 >= mip_data_size[1]:
                                edx[1,1] = 0
                            if tilez1 == 0:
                                edx[2,0] = 0
                            if tilez2 >= mip_data_size[2]:
                                edx[2,1] = 0
                        else:
                            edx = np.zeros((3,2), dtype=int)
                        
                        if params['slicestep'] == 1:
                            seg_image, values, counts, bboxes = vast.get_seg_image_rle_decoded_bboxes(params['miplevel'], tilex1-int(edx[0,0]),tilex2+int(edx[0,1]),tiley1-int(edx[1,0]),tiley2+int(edx[1,1]),tilez1-int(edx[2,0]),tilez2+int(edx[2,1]), False)
                        else:
                            bs = blockslicenumbers[tz]
                            if edx[2,0] == 1:
                                bs = [bs[0] - params['slicestep']] + bs
                            if edx[2, 1] == 1:  
                                bs = bs + [bs[-1] + params['slicestep']]
                            
                            seg_image = np.zeros((tilex2 - tilex1 + 1 + int(edx[0,0]) + int(edx[0,1]),
                                                   tiley2 - tiley1 + 1 + int(edx[1,0]) + int(edx[1,1]),
                                                   len(bs)), dtype=np.uint16)
                            numarr = np.zeros(max_object_number, dtype=np.int32)
                            bboxarr = np.full((max_object_number, 6), -1, dtype=float)
                            firstblockslice = bs[0]

                            for i, slice_z in enumerate(bs):
                                ssegimage, svalues, snumbers, sbboxes = vast.get_seg_image_rle_decoded_bboxes(
                                    params['miplevel'], 
                                    tilex1 - int(edx[0,0]), tilex2 + int(edx[0,1]),
                                    tiley1 - int(edx[1,0]), tiley2 + int(edx[1,1]),
                                    slice_z, slice_z, False)
                                
                                seg_image[:,:,i] = ssegimage
                                
                                # Update counts and bboxes
                                if len(svalues) > 0:
                                    # Filter out background (0)
                                    mask = svalues != 0
                                    svalues = svalues[mask]
                                    snumbers = snumbers[mask]
                                    sbboxes = sbboxes[mask, :]
                                    
                                    # Adjust Z coordinates in bboxes
                                    sbboxes[:, [2, 5]] = sbboxes[:, [2, 5]] + (i - 1)
                                    
                                    # Update arrays
                                    numarr[svalues - 1] += snumbers  # svalues are 1-based
                                    
                                    # Expand bounding boxes (merge overlapping regions)
                                    for val_idx, val in enumerate(svalues):
                                        if bboxarr[val - 1, 2] == -1:  # First occurrence
                                            bboxarr[val - 1] = sbboxes[val_idx]
                                        else:
                                            # Merge boxes
                                            bboxarr[val - 1, 0] = min(bboxarr[val - 1, 0], sbboxes[val_idx, 0])
                                            bboxarr[val - 1, 1] = min(bboxarr[val - 1, 1], sbboxes[val_idx, 1])
                                            bboxarr[val - 1, 2] = min(bboxarr[val - 1, 2], sbboxes[val_idx, 2])
                                            bboxarr[val - 1, 3] = max(bboxarr[val - 1, 3], sbboxes[val_idx, 3])
                                            bboxarr[val - 1, 4] = max(bboxarr[val - 1, 4], sbboxes[val_idx, 4])
                                            bboxarr[val - 1, 5] = max(bboxarr[val - 1, 5], sbboxes[val_idx, 5])
                            # Extract non-zero values
                            values = np.where(numarr > 0)[0] + 1  # Convert back to 1-based
                            counts = numarr[values - 1]
                            bboxes = bboxarr[values - 1]
                        
                        if params['erodedilate']:
                            # Erode then dilate to remove 1-voxel-thin objects
                            struct_element = np.ones((2,2,2), dtype=bool)
                            seg_image = ndimage.binary_opening(seg_image, structure=struct_element)
                            # Crop padding
                            if seg_image.ndim == 3:
                                seg_image = seg_image[int(edx[0,0]):seg_image.shape[0]-int(edx[0,1]),int(edx[1,0]):seg_image.shape[1]-int(edx[1,1]),int(edx[2,0]):seg_image.shape[2]-int(edx[2,1])]
                            # Adjust bounding boxes if they exist
                            if bboxes.size > 0 and bboxes.shape[0] > 0:
                                # Adjust X coordinates (columns 0 and 3, 0-indexed)
                                if edx[0,0] > 0:
                                    bb = bboxes[:, [0, 3]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [0, 3]] = bb
                                
                                if edx[0,1] > 0:
                                    bb = bboxes[:, [0, 3]]
                                    bb[bb > seg_image.shape[0]] = seg_image.shape[0]
                                    bboxes[:, [0, 3]] = bb
                                
                                # Adjust Y coordinates (columns 1 and 4, 0-indexed)
                                if edx[1,0] > 0:
                                    bb = bboxes[:, [1, 4]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [1, 4]] = bb
                                
                                if edx[1,1] > 0:
                                    bb = bboxes[:, [1, 4]]
                                    bb[bb > seg_image.shape[1]] = seg_image.shape[1]
                                    bboxes[:, [1, 4]] = bb
                                
                                # Adjust Z coordinates (columns 2 and 5, 0-indexed)
                                if edx[2,0] > 0:
                                    bb = bboxes[:, [2, 5]] - 1
                                    bb[bb == 0] = 1
                                    bboxes[:, [2, 5]] = bb
                                
                                if edx[2,1] > 0:
                                    bb = bboxes[:, [2, 5]]
                                    bb[bb > seg_image.shape[2]] = seg_image.shape[2]
                                    bboxes[:, [2, 5]] = bb

                else:
                    if not params['usemipregionconstraint'] or mc_loadflags[tx, ty, tz]:
                        # Read this cube
                        if params['slicestep'] == 1:
                            scsimage = vast.get_screenshot_image(params['miplevel'], tilex1, tilex2, tiley1, tiley2, tilez1, tilez2, True, True)
                            if tilez1 == tilez2:
                                # in this case a 3d array will be returned, but a 4d array with singular dimension 3 is expected below.
                                scsimage = scsimage.reshape(scsimage.shape[0], scsimage.shape[1], 1, scsimage.shape[2])
                        else:
                            bs = blockslicenumbers[tz]
                            scsimage = np.zeros((tiley2 - tiley1 + 1,
                                                    tilex2 - tilex1 + 1,
                                                    len(bs),
                                                    3), dtype=np.uint8)
                            firstblockslice = bs[0]
                            for i, slice_z in enumerate(bs):
                                scslice = vast.get_screenshot_image(
                                    params['miplevel'],
                                    tilex1, tilex2,
                                    tiley1, tiley2,
                                    slice_z, slice_z,
                                    True, True)
                                if scslice is not None:
                                    if scslice.ndim == 3:
                                        scslice = scslice.reshape(scslice.shape[0], scslice.shape[1], 1, scslice.shape[2])
                                    scsimage[:, :, i, :] = scslice
                                else:
                                    scsimage[:, :, i, :] = np.zeros((tiley2 - tiley1 + 1, tilex2 - tilex1 + 1, 1, 3), dtype=np.uint8)

                if extract_seg:
                    if not params['usemipregionconstraint'] or mc_loadflags[tx, ty, tz]:
                        print(f"Processing Segmentation Cubes {tx,ty,tz} of {nrxtiles,nrytiles,nrztiles}...")
                        
                        mask = values != 0
                        counts_f = counts[mask]
                        bboxes_f = bboxes[mask]
                        values_f = values[mask]

                        if len(values) > 0:
                            #  VAST translates voxel data before transmission 
                            xvofs = 0
                            yvofs = 0
                            zvofs = 0
                            ttxs = tilexs
                            ttys = tileys
                            ttzs = tilezs

                            # Close surfaces
                            if params['closesurfaces']:
                                # Add empty slice at start X
                                if tx == 1:
                                    segimage = np.concatenate([np.zeros((1, segimage.shape[1], segimage.shape[2]), dtype=segimage.dtype),
                                    segimage], axis=0)
                                    bboxes_f[:, 0] += 1 # xmin
                                    bboxes_f[:, 3] += 1 # xmax
                                    xvofs -= 1 
                                    ttxs += 1
                                
                                # Add empty slice at start Y
                                if ty == 1:
                                    segimage = np.concatenate([np.zeros((segimage.shape[0], 1, segimage.shape[2]), dtype=segimage.dtype), segimage], axis=1)
                                    bboxes_f[:, 1] += 1  # ymin
                                    bboxes_f[:, 4] += 1  # ymax
                                    yvofs -= 1
                                    ttys += 1
                                
                                # Add empty slice at start Z
                                if tz == 1:
                                    segimage = np.concatenate([np.zeros((segimage.shape[0], segimage.shape[1], 1), dtype=segimage.dtype), segimage], axis=2)
                                    bboxes_f[:, 2] += 1  # zmin
                                    bboxes_f[:, 5] += 1  # zmax
                                    zvofs -= 1
                                    ttzs += 1

                                # Add empty slice at end X
                                if tx == nrxtiles:
                                    segimage = np.concatenate([
                                        segimage,
                                        np.zeros((1, segimage.shape[1], segimage.shape[2]), dtype=segimage.dtype)
                                    ], axis=0)
                                    ttxs += 1
                                
                                # Add empty slice at end Y
                                if ty == nrytiles:
                                    segimage = np.concatenate([
                                        segimage,
                                        np.zeros((segimage.shape[0], 1, segimage.shape[2]), dtype=segimage.dtype)
                                    ], axis=1)
                                    ttys += 1
                                
                                # Add empty slice at end Z
                                if tz == nrztiles:
                                    segimage = np.concatenate([
                                        segimage,
                                        np.zeros((segimage.shape[0], segimage.shape[1], 1), dtype=segimage.dtype)
                                    ], axis=2)
                                    ttzs += 1

                            for segnr in range(len(values_f)):
                                # Progress message every 10 segments
                                if segnr % 10 == 0:
                                    seg_progress = f"Objects {segnr+1}-{min(segnr+10, len(values_f))} of {len(values_f)} ..."
                                    print(f"  {seg_progress}")
                                
                                seg = values_f[segnr]
                                bbx = bboxes_f[segnr].copy()
                                
                                # Expand bounding box by 1 pixel in each direction
                                bbx += np.array([-1, -1, -1, 1, 1, 1])
                                
                                # Clamp to volume bounds (1-based like MATLAB)
                                bbx[0] = max(bbx[0], 1)
                                bbx[1] = max(bbx[1], 1)
                                bbx[2] = max(bbx[2], 1)
                                bbx[3] = min(bbx[3], ttxs)
                                bbx[4] = min(bbx[4], ttys)
                                bbx[5] = min(bbx[5], ttzs)
                                
                                # Ensure at least 2 pixels in each direction
                                if bbx[0] == bbx[3]:
                                    if bbx[0] > 1:
                                        bbx[0] -= 1
                                    else:
                                        bbx[3] += 1
                                
                                if bbx[1] == bbx[4]:
                                    if bbx[1] > 1:
                                        bbx[1] -= 1
                                    else:
                                        bbx[4] += 1
                                
                                if bbx[2] == bbx[5]:
                                    if bbx[2] > 1:
                                        bbx[2] -= 1
                                    else:
                                        bbx[5] += 1
                                
                                # Extract subvolume (convert from 1-based to 0-based indexing)
                                subseg = segimage[
                                    bbx[0]-1:bbx[3],
                                    bbx[1]-1:bbx[4],
                                    bbx[2]-1:bbx[5]
                                ]
                                
                                # Create binary mask
                                subseg = (subseg == seg).astype(float)
                                
                                # Run marching cubes
                                from skimage import measure
                                verts, faces, normals, values_mc = measure.marching_cubes(subseg, level=0.5)
                                
                                if len(verts) > 0:
                                    # Adjust coordinates for bounding box and added empty slices
                                    # Note: MATLAB isosurface returns [X, Y, Z], scikit-image returns [Y, X, Z]
                                    # So we need to swap columns
                                    verts_adjusted = verts.copy()
                                    
                                    # Swap from (Y, X, Z) to (X, Y, Z)
                                    verts_adjusted[:, 0], verts_adjusted[:, 1] = verts[:, 1], verts[:, 0]
                                    
                                    # Adjust for bbox (MATLAB 1-based)
                                    verts_adjusted[:, 0] += bbx[1] - 1 + yvofs  # X (was Y in MATLAB indexing)
                                    verts_adjusted[:, 1] += bbx[0] - 1 + xvofs  # Y (was X in MATLAB indexing)
                                    verts_adjusted[:, 2] += bbx[2] - 1 + zvofs  # Z
                                    
                                    # Adjust for tile position
                                    verts_adjusted[:, 0] += tiley1 - 1
                                    verts_adjusted[:, 1] += tilex1 - 1
                                    
                                    if params['slicestep'] == 1:
                                        verts_adjusted[:, 2] += tilez1 - 1
                                    else:
                                        verts_adjusted[:, 2] = ((verts_adjusted[:, 2] - 0.5) * params['slicestep']) + 0.5 + firstblockslice - 1
                                    
                                    # Scale to physical units (nm)
                                    verts_adjusted[:, 0] *= params['yscale'] * params['yunit'] * mipfacty
                                    verts_adjusted[:, 1] *= params['xscale'] * params['xunit'] * mipfactx
                                    verts_adjusted[:, 2] *= params['zscale'] * params['zunit'] * mipfactz
                                    
                                    # Store in arrays
                                    params['farray'][(seg, tx, ty, tz)] = faces
                                    params['varray'][(seg, tx, ty, tz)] = verts_adjusted

                else:
                    if not params['usemipregionconstraint'] or mc_loadflags[tx, ty, tz]:
                        print(f"Processing Screenshot Cubes ({tx},{ty},{tz}) of ({nrxtiles},{nrytiles},{nrztiles})...")
                        
                        # Extract RGB channels from screenshot cube
                        # scsimage shape: (tiley2-tiley1+1, tilex2-tilex1+1, num_z, 3)
                        rcube = np.squeeze(scsimage[:, :, :, 0]).transpose(1, 0, 2)  # Transpose to match MATLAB [2 1 3]
                        gcube = np.squeeze(scsimage[:, :, :, 1]).transpose(1, 0, 2)
                        bcube = np.squeeze(scsimage[:, :, :, 2]).transpose(1, 0, 2)
                        
                        # Ensure 3D arrays
                        if rcube.ndim == 2:
                            rcube = rcube[:, :, np.newaxis]
                        if gcube.ndim == 2:
                            gcube = gcube[:, :, np.newaxis]
                        if bcube.ndim == 2:
                            bcube = bcube[:, :, np.newaxis]
                        
                        # Initialize offset variables
                        xvofs = 0
                        yvofs = 0
                        zvofs = 0
                        ttxs = tilexs
                        ttys = tileys
                        ttzs = tilezs
                        
                        # Close surfaces - add empty slices at boundaries
                        if params['closesurfaces']:
                            # Add empty slice at start X
                            if tx == 1:
                                rcube = np.concatenate([np.zeros((1, rcube.shape[1], rcube.shape[2]), dtype=rcube.dtype), rcube], axis=0)
                                gcube = np.concatenate([np.zeros((1, gcube.shape[1], gcube.shape[2]), dtype=gcube.dtype), gcube], axis=0)
                                bcube = np.concatenate([np.zeros((1, bcube.shape[1], bcube.shape[2]), dtype=bcube.dtype), bcube], axis=0)
                                xvofs = -1
                                ttxs += 1
                            
                            # Add empty slice at start Y
                            if ty == 1:
                                rcube = np.concatenate([np.zeros((rcube.shape[0], 1, rcube.shape[2]), dtype=rcube.dtype), rcube], axis=1)
                                gcube = np.concatenate([np.zeros((gcube.shape[0], 1, gcube.shape[2]), dtype=gcube.dtype), gcube], axis=1)
                                bcube = np.concatenate([np.zeros((bcube.shape[0], 1, bcube.shape[2]), dtype=bcube.dtype), bcube], axis=1)
                                yvofs = -1
                                ttys += 1
                            
                            # Add empty slice at start Z
                            if tz == 1:
                                rcube = np.concatenate([np.zeros((rcube.shape[0], rcube.shape[1], 1), dtype=rcube.dtype), rcube], axis=2)
                                gcube = np.concatenate([np.zeros((gcube.shape[0], gcube.shape[1], 1), dtype=gcube.dtype), gcube], axis=2)
                                bcube = np.concatenate([np.zeros((bcube.shape[0], bcube.shape[1], 1), dtype=bcube.dtype), bcube], axis=2)
                                zvofs = -1
                                ttzs += 1
                            
                            # Add empty slice at end X
                            if tx == nrxtiles:
                                rcube = np.concatenate([rcube, np.zeros((1, rcube.shape[1], rcube.shape[2]), dtype=rcube.dtype)], axis=0)
                                gcube = np.concatenate([gcube, np.zeros((1, gcube.shape[1], gcube.shape[2]), dtype=gcube.dtype)], axis=0)
                                bcube = np.concatenate([bcube, np.zeros((1, bcube.shape[1], bcube.shape[2]), dtype=bcube.dtype)], axis=0)
                                ttxs += 1
                            
                            # Add empty slice at end Y
                            if ty == nrytiles:
                                rcube = np.concatenate([rcube, np.zeros((rcube.shape[0], 1, rcube.shape[2]), dtype=rcube.dtype)], axis=1)
                                gcube = np.concatenate([gcube, np.zeros((gcube.shape[0], 1, gcube.shape[2]), dtype=gcube.dtype)], axis=1)
                                bcube = np.concatenate([bcube, np.zeros((bcube.shape[0], 1, bcube.shape[2]), dtype=bcube.dtype)], axis=1)
                                ttys += 1
                            
                            # Add empty slice at end Z
                            if tz == nrztiles:
                                rcube = np.concatenate([rcube, np.zeros((rcube.shape[0], rcube.shape[1], 1), dtype=rcube.dtype)], axis=2)
                                gcube = np.concatenate([gcube, np.zeros((gcube.shape[0], gcube.shape[1], 1), dtype=gcube.dtype)], axis=2)
                                bcube = np.concatenate([bcube, np.zeros((bcube.shape[0], bcube.shape[1], 1), dtype=bcube.dtype)], axis=2)
                                ttzs += 1
                        
                        # Extract isosurfaces
                        if params['extractwhich'] == 10:
                            # Extract unique colors from screenshots as individual 3d objects
                            # Combine RGB channels into single color value
                            colcube = (rcube.astype(np.int32) << 16) + (gcube.astype(np.int32) << 8) + bcube.astype(np.int32)
                            
                            # Get unique colors and their counts
                            unique_colors, inverse_indices = np.unique(colcube, return_inverse=True)
                            num_pixels = np.bincount(inverse_indices)
                            
                            # Remove background (color 0)
                            if unique_colors[0] == 0:
                                unique_colors = unique_colors[1:]
                                num_pixels = num_pixels[1:]
                            
                            # Update object volume
                            params['object_volume'][unique_colors] += num_pixels[:, np.newaxis]
                            
                            if len(unique_colors) > 0:
                                colnr = 0
                                
                                while colnr < len(unique_colors) and True:  # vdata.state.lastcancel==0
                                    endcolnr = len(unique_colors)  # Process all colors in one batch
                                    
                                    # Process colors in batch
                                    for pcolnr in range(colnr, endcolnr):
                                        color_val = unique_colors[pcolnr]
                                        
                                        # Create binary mask for this color
                                        psubseg = (colcube == color_val).astype(float)
                                        
                                        # Extract isosurface
                                        if psubseg.sum() > 0:  # Only if color exists in volume
                                            verts, faces, _, _ = measure.marching_cubes(psubseg, level=0.5)
                                            
                                            if len(verts) > 0:
                                                # Adjust coordinates for bbox and added empty slices
                                                # Swap axes from (Y, X, Z) to (X, Y, Z)
                                                verts_adjusted = verts.copy()
                                                verts_adjusted[:, 0], verts_adjusted[:, 1] = verts[:, 1], verts[:, 0]
                                                
                                                # Apply offsets from closed surfaces
                                                verts_adjusted[:, 0] += yvofs
                                                verts_adjusted[:, 1] += xvofs
                                                verts_adjusted[:, 2] += zvofs
                                                
                                                # Adjust for tile boundaries
                                                verts_adjusted[:, 0] += tiley1 - 1
                                                verts_adjusted[:, 1] += tilex1 - 1
                                                
                                                if params['slicestep'] == 1:
                                                    verts_adjusted[:, 2] += tilez1 - 1
                                                else:
                                                    verts_adjusted[:, 2] = ((verts_adjusted[:, 2] - 0.5) * params['slicestep']) + 0.5 + firstblockslice - 1
                                                
                                                # Scale to physical units (nm)
                                                verts_adjusted[:, 0] *= params['yscale'] * params['yunit'] * mipfacty
                                                verts_adjusted[:, 1] *= params['xscale'] * params['xunit'] * mipfactx
                                                verts_adjusted[:, 2] *= params['zscale'] * params['zunit'] * mipfactz
                                                
                                                # Store mesh data
                                                idx = tz * params['nrytiles'] * params['nrxtiles'] + ty * params['nrxtiles'] + tx
                                                params['fvindex'][color_val, idx] = block_nr
                                                params['farray'].append(faces)
                                                params['varray'].append(verts_adjusted)
                                                block_nr += 1
                                    
                                    colnr = endcolnr + 1
                        
                        else:
                            # Extract by brightness threshold
                            if params['extractwhich'] in [6, 7, 8, 9]:
                                cube = (rcube.astype(int) + gcube.astype(int) + bcube.astype(int)) // 3
                                cube = cube.astype(np.uint8)
                            
                            # Process each object
                            for obj_idx in range(len(names)):
                                if params['extractwhich'] == 5:
                                    # RGB channels
                                    if obj_idx == 0:
                                        subseg = (rcube > 128).astype(float)
                                    elif obj_idx == 1:
                                        subseg = (gcube > 128).astype(float)
                                    else:
                                        subseg = (bcube > 128).astype(float)
                                
                                elif params['extractwhich'] in [6, 7, 8, 9]:
                                    # Brightness threshold
                                    threshold = params['lev'][obj_idx] if obj_idx < len(params['lev']) else 128
                                    subseg = (cube > threshold).astype(float)
                                
                                # Extract isosurface
                                if subseg.ndim == 3 and subseg.sum() > 0:
                                    verts, faces, _, _ = measure.marching_cubes(subseg, level=0.5)
                                    
                                    if len(verts) > 0:
                                        # Adjust coordinates
                                        verts_adjusted = verts.copy()
                                        verts_adjusted[:, 0], verts_adjusted[:, 1] = verts[:, 1], verts[:, 0]
                                        
                                        # Apply offsets
                                        verts_adjusted[:, 0] += yvofs
                                        verts_adjusted[:, 1] += xvofs
                                        verts_adjusted[:, 2] += zvofs
                                        
                                        # Adjust for tile boundaries
                                        verts_adjusted[:, 0] += tiley1 - 1
                                        verts_adjusted[:, 1] += tilex1 - 1
                                        
                                        if params['slicestep'] == 1:
                                            verts_adjusted[:, 2] += tilez1 - 1
                                        else:
                                            verts_adjusted[:, 2] = ((verts_adjusted[:, 2] - 0.5) * params['slicestep']) + 0.5 + firstblockslice - 1
                                        
                                        # Scale to physical units
                                        verts_adjusted[:, 0] *= params['yscale'] * params['yunit'] * mipfacty
                                        verts_adjusted[:, 1] *= params['xscale'] * params['xunit'] * mipfactx
                                        verts_adjusted[:, 2] *= params['zscale'] * params['zunit'] * mipfactz
                                        
                                        # Store mesh data
                                        params['farray'][obj_idx, tx, ty, tz] = faces
                                        params['varray'][obj_idx, tx, ty, tz] = verts_adjusted

                tilex1 = tilex1 + params['blocksizex'] - params['overlap']
                tx += 1
            tiley1 = tiley1 + params['blocksizey'] - params['overlap']
            ty += 1
        tilez1 = tilez1 + params['blocksizez'] - params['overlap']
        tz += 1
    
    # Mesh merging phase
    if True:  # Check that export wasn't cancelled
        print("Exporting Surfaces... Merging meshes...")
        
        # Initialize surface area array
        object_surface_area = np.zeros((len(params['objects']) if params['objects'] is not None else 0, 1))
        
        # Determine colors for objects
        if extract_seg:
            # Get colors from segments
            colors = np.zeros((max_object_number + 1, 3), dtype=np.uint8)
            if params['objects'] is not None:
                for idx, (seg_id, flags) in enumerate(params['objects']):
                    # Find segment in all_seg_data to get its color
                    seg_data = next((s for s in all_seg_data if s['id'] == seg_id), None)
                    if seg_data:
                        colors[seg_id] = seg_data.get('color', [255, 255, 255])
        else:
            # Colors for screenshot modes
            if params['extractwhich'] == 5:
                # RGB channels
                colors = np.zeros((3, 3), dtype=np.uint8)
                colors[0, 0] = 255  # Red
                colors[1, 1] = 255  # Green
                colors[2, 2] = 255  # Blue
            elif params['extractwhich'] in [6, 7, 8, 9]:
                # Brightness thresholds - use grayscale
                colors = np.zeros((len(params['lev']), 3), dtype=np.uint8)
                for i, lev in enumerate(params['lev']):
                    colors[i] = [lev, lev, lev]
            elif params['extractwhich'] == 10:
                # Colors are the color values themselves
                colors = None  # Will be generated from color values
        
        # Merge and write objects
        if params['extractwhich'] == 10:
            # Find which colors actually have data
            params['objects'] = np.where(params['object_volume'] > 0)[0]
        
        if params['objects'] is not None:
            num_objects = len(params['objects']) if isinstance(params['objects'], (list, np.ndarray)) else params['objects'].shape[0]
        else:
            num_objects = 0
        
        for obj_idx in range(num_objects):
            if params['objects'] is not None:
                if isinstance(params['objects'], np.ndarray) and params['objects'].ndim > 1:
                    seg_id = params['objects'][obj_idx, 0]
                else:
                    seg_id = params['objects'][obj_idx]
            else:
                continue
            
            # Create segment name
            if params['extractwhich'] == 10:
                # Color-based name
                r = (seg_id >> 16) & 0xFF
                g = (seg_id >> 8) & 0xFF
                b = seg_id & 0xFF
                seg_name = f'col_{r:02X}{g:02X}{b:02X}'
            else:
                # Use object name from names list
                if obj_idx < len(names):
                    seg_name = names[obj_idx]
                else:
                    seg_name = f'Object_{seg_id}'
            
            print(f"Exporting Surfaces... Merging parts of {seg_name}...")
            
            # Initialize merged mesh
            cofp = []
            covp = []
            vofs = 0
            
            # Merge across Z tiles
            for tz in range(params['nrztiles']):
                # Merge across Y tiles
                for ty in range(params['nrytiles']):
                    # Merge across X tiles
                    for tx in range(params['nrxtiles']):
                        if params['extractwhich'] == 10:
                            # Look up in fvindex sparse matrix
                            idx = tz * params['nrytiles'] * params['nrxtiles'] + ty * params['nrxtiles'] + tx
                            blocknr = int(params['fvindex'][seg_id, idx]) if params['fvindex'][seg_id, idx] else 0
                            
                            if blocknr == 0:
                                if tx == 0:
                                    f = np.empty((0, 3), dtype=np.int32)
                                    v = np.empty((0, 3), dtype=np.float32)
                            else:
                                if tx == 0:
                                    f = params['farray'][blocknr - 1].copy() if isinstance(params['farray'][blocknr - 1], np.ndarray) else np.empty((0, 3), dtype=np.int32)
                                    v = params['varray'][blocknr - 1].copy() if isinstance(params['varray'][blocknr - 1], np.ndarray) else np.empty((0, 3), dtype=np.float32)
                                else:
                                    f, v = merge_meshes(f, v, params['farray'][blocknr - 1], params['varray'][blocknr - 1])
                        else:
                            # Segmentation mode - use 3D indexing
                            if tx == 0:
                                f = params['farray'][seg_id, tx, ty, tz]
                                v = params['varray'][seg_id, tx, ty, tz]
                            else:
                                f, v = merge_meshes(f, v, params['farray'][seg_id, tx, ty, tz], params['varray'][seg_id, tx, ty, tz])
                    
                    # Merge rows (Y direction)
                    if ty == 0:
                        fc = f.copy() if isinstance(f, np.ndarray) else np.empty((0, 3), dtype=np.int32)
                        vc = v.copy() if isinstance(v, np.ndarray) else np.empty((0, 3), dtype=np.float32)
                    else:
                        fc, vc = merge_meshes(fc, vc, f, v)
                
                # Merge planes (Z direction)
                if tz == 0:
                    fp = fc.copy() if isinstance(fc, np.ndarray) else np.empty((0, 3), dtype=np.int32)
                    vp = vc.copy() if isinstance(vc, np.ndarray) else np.empty((0, 3), dtype=np.float32)
                else:
                    fp, vp = merge_meshes(fp, vp, fc, vc)
                    
                    # Remove non-overlapping parts to speed up computation
                    if vp.shape[0] > 1 and fp.shape[0] > 1:
                        # Find the z-coordinate boundary
                        vcut = np.where(vp[:, 2] == np.max(vp[:, 2]))[0][0] - 1
                        
                        # Find indices where faces reference vertices > vcut
                        mask = np.any(fp > vcut, axis=1)
                        if np.any(mask):
                            fcutind = np.where(mask)[0][0]
                            fcut = fcutind - 1
                            
                            # Move old vertices and faces to storage
                            if vcut >= 0:
                                covp.append(vp[:vcut+1, :])
                                vp = vp[vcut+1:, :]
                            
                            ovofs = vofs
                            vofs = vofs + vcut + 1
                            
                            if fcut >= 0:
                                fp_adjusted = fp[:fcut+1, :] + ovofs
                                cofp.append(fp_adjusted)
                                fp = fp[fcut+1:, :] - (vcut + 1)
            
            # Combine stored and current vertices/faces
            if len(covp) > 0:
                vp = np.vstack([np.vstack(covp), vp])
            if len(cofp) > 0:
                fp = np.vstack([np.vstack(cofp), fp + vofs])
            
            # Invert Z axis if requested
            if params['invertz'] and vp.shape[0] > 0:
                vp[:, 2] = -vp[:, 2]
            
            # Add offset if requested
            if params['outputoffsetx'] != 0 and vp.shape[0] > 0:
                vp[:, 0] += params['outputoffsetx']
            if params['outputoffsety'] != 0 and vp.shape[0] > 0:
                vp[:, 1] += params['outputoffsety']
            if params['outputoffsetz'] != 0 and vp.shape[0] > 0:
                vp[:, 2] += params['outputoffsetz']
            
            # Sanitize segment name (replace invalid characters)
            seg_name_clean = seg_name.replace(' ', '_').replace('?', '_').replace('*', '_')
            seg_name_clean = seg_name_clean.replace('\\', '_').replace('/', '_').replace('|', '_')
            seg_name_clean = seg_name_clean.replace(':', '_').replace('"', '_').replace('<', '_').replace('>', '_')
            
            # Save model if it has geometry
            if not params.get('skipmodelgeneration', False) and vp.shape[0] > 0:
                output_folder = params.get('targetfolder', './vast_export')
                output_prefix = params.get('targetfileprefix', 'Segment_')
                
                # Ensure output folder exists
                import os
                os.makedirs(output_folder, exist_ok=True)
                
                # Save as OBJ format
                filename = os.path.join(output_folder, f"{output_prefix}_{seg_id:04d}_{seg_name_clean}.obj")
                mtl_filename = os.path.join(output_folder, f"{output_prefix}_{seg_id:04d}_{seg_name_clean}.mtl")
                object_name = f"{output_prefix}_{seg_id:04d}_{seg_name_clean}"
                material_name = f"{output_prefix}_{seg_id:04d}_material"
                
                print(f"Saving {filename} as Wavefront OBJ...")
                
                # Save OBJ file
                save_obj(filename, vp, fp)
                
                # Save MTL file
                if params['extractwhich'] == 10:
                    col = np.array([(seg_id >> 16) & 0xFF, (seg_id >> 8) & 0xFF, seg_id & 0xFF]) / 255.0
                else:
                    col = colors[seg_id] / 255.0 if seg_id < colors.shape[0] else np.array([1, 1, 1])
                
                save_material_file(mtl_filename, material_name, col, 1.0)

            
            # Compute surface area if requested
            if params.get('savesurfacestats', False) and vp.shape[0] > 0:
                surface_area = 0.0
                for tri_idx in range(fp.shape[0]):
                    v0 = vp[fp[tri_idx, 0]]
                    v1 = vp[fp[tri_idx, 1]]
                    v2 = vp[fp[tri_idx, 2]]
                    
                    # Compute cross product and area
                    cross = np.cross(v1 - v0, v2 - v0)
                    surface_area += np.linalg.norm(cross) / 2.0
                
                object_surface_area[obj_idx] = surface_area
    
    if extract_seg:
        vast.set_seg_translation([], [])
    
    # last cancel == 0, doesnt exist in this case so continue inside
    if extract_seg:
        params['object_surface_area'] = np.zeros((objects.shape[0], 1))
        match params['object_colors']:
            case 1:
                num_objects = params['objects'].shape[0]
                colors = np.zeros((num_objects, 3))
                for segnr in range(num_objects):
                   seg = int(params['objects'][segnr, 0]) - 1
                   inheritseg = int(data[seg, 17]) - 1
                   colors[seg, :] = data[inheritseg, 2:5]
            case 2:
                cmap_jet = plt.get_cmap('jet', 256)
        
                # Calculate volume proportions
                vol_data = vdata['data']['measurevol']['lastvolume']
                vols = (255 * vol_data / np.max(vol_data)).astype(int)
                
                # Get colors from colormap (returns 0-1 range)
                cols = cmap_jet(vols)[:, :3] 
                
                objs = vdata['data']['measurevol']['lastobjects'][:, 0].astype(int) - 1
                
                colors = np.zeros((params['objects'].shape[0], 3))
                colors[objs, :] = cols * 255
    else:
        if params['extractwhich'] == 5:
            # Create an Nx3 array of zeros
            colors = np.zeros((params['objects'].shape[0], 3))
            # MATLAB 1-based indexing (1,1), (2,2), (3,3) 
            # converts to Python 0-based indexing [0,0], [1,1], [2,2]
            colors[0, 0] = 255
            colors[1, 1] = 255
            colors[2, 2] = 255 

        elif params['extractwhich'] in [6, 7, 8, 9]:
            lev = params['lev'].flatten() # Ensure it's a 1D vector
            colors = np.column_stack((lev, lev, lev))
                
        elif params['extractwhich'] == 10:
            # IDs are colors in this case 
            pass

    # Write 3dsmax bulk loader script? vdata doesnt exist, it's used in the gui context
    # if vdata['data']['exportobj']['write3dsmaxloader'] == 1:
    #     # Assuming save3dsmaxloader is a defined function
    #     save3dsmaxloader(f"{params['targetfolder']}loadallobj_here.ms")
    
    # merge full objects from components
    if params['extractwhich'] == 10:
        # np.where returns indices where condition is true; +1 if you need to keep 1-based IDs
        indices = np.where(params['object_volume'] > 0)
        params['objects'] = indices.reshape(-1, 1)

    # 3. Loop through objects
    segnr = 0  # Python uses 0-based indexing
    num_objects = params['objects'].shape

    
    while segnr < num_objects:
        seg = int(params['objects'][segnr, 0])
        
        if params['extractwhich'] == 10:
            print(f"Exporting Surfaces ... Merging parts of object {segnr + 1} / {num_objects}...")
            
            # Bitwise operations for hex color string
            r = (seg >> 16) & 255
            g = (seg >> 8) & 255
            b = seg & 255
            segname = f"col_{r:02X}{g:02X}{b:02X}"
        else:
            # Assuming 'name' is a list of strings
            # We subtract 1 from seg if it contains 1-based MATLAB indices
            current_name = name[seg - 1]
            print(f"Exporting Surfaces ... Merging parts of {current_name}...")
            segname = current_name

        # MATLAB: pause(0.01)
        time.sleep(0.01)
        
        # Initialize variables for the next steps
        cofp = []
        covp = []
        vofs = 0
        
        segnr += 1

        z = 1
        while z < params['nrztiles']:
            y = 0
            while y < params['nrytiles']:
                x = 0
                while x < params['nrxtiles']:
                    if params['extractwhich'] == 10:
                        # MATLAB index calculation: z*nry*nrx + y*nrx + x
                        idx = z * params['nrytiles'] * params['nrxtiles'] + y * params['nrxtiles'] + x
                        # MATLAB full() equivalent for sparse or dense array access
                        blocknr = int(params['fvindex'][seg, idx])
                        
                        if blocknr == 0:
                            if x == 0:
                                f, v = [], []
                        else:
                            if x == 0:
                                f = params['farray'][blocknr - 1] # -1 for 0-based indexing
                                v = params['varray'][blocknr - 1]
                            else:
                                # mergemeshes must be a user-defined function in your environment
                                f, v = mergemeshes(f, v, params['farray'][blocknr - 1], params['varray'][blocknr - 1])
                    else:
                        # Case where extractwhich != 10 (Cell array access)
                        if x == 0:
                            f = params['farray'][seg, x, y, z]
                            v = params['varray'][seg, x, y, z]
                        else:
                            f, v = mergemeshes(f, v, params['farray'][seg, x, y, z], params['varray'][seg, x, y, z])
                    x += 1
                    
                # Row merging logic
                if y == 0:
                    fc, vc = f, v
                else:
                    fc, vc = mergemeshes(fc, vc, f, v)
                y += 1
                
            # Plane merging and optimization logic
            if z == 0:
                fp, vp = fc, vc
            else:
                fp, vp = mergemeshes(fp, vp, fc, vc)
                
                # Optimization: Take out non-overlapping parts
                if len(vp) > 1 and len(fp) > 1:
                    # MATLAB: find(vp(:,3)==max(vp(:,3)),1,'first')-1
                    max_z = np.max(vp[:, 2])
                    vcut = np.where(vp[:, 2] == max_z)[0][0] # first occurrence
                    
                    # MATLAB: find(fp > vcut, 1, 'first')
                    fcutind = np.where(fp > vcut)[0][0]
                    
                    # MATLAB: ind2sub(size(fp), fcutind) -> Python: unravel_index
                    fcut, _ = np.unravel_index(fcutind, fp.shape)
                    
                    # Accumulate results
                    covp = np.vstack([covp, vp[:vcut, :]]) if 'covp' in locals() else vp[:vcut, :]
                    vp = vp[vcut:, :]
                    
                    ovofs = vofs
                    vofs += vcut
                    
                    cofp = np.vstack([cofp, fp[:fcut, :] + ovofs]) if 'cofp' in locals() else fp[:fcut, :] + ovofs
                    fp = fp[fcut:, :] - vcut
                    
            z += 1
        
        if len(covp) > 0:
            vp = np.vstack([covp, vp])
            fp = np.vstack([cofp, fp + vofs])

        # Invert Z axis if requested
        if params['invertz']:
            if vp.shape[0] > 0:
                vp[:, 2] = -vp[:, 2]
        
        # Apply output offsets if specified
        if params['outputoffsetx'] != 0:
            if vp.shape[0] > 0:
                vp[:, 0] += params['outputoffsetx']
        if params['outputoffsety'] != 0:
            if vp.shape[0] > 0:
                vp[:, 1] += params['outputoffsety']
        if params['outputoffsetz'] != 0:
            if vp.shape[0] > 0:
                vp[:, 2] += params['outputoffsetz']
        
        # Sanitize segment name
        if extract_seg:
            idx = np.where(data[:, 0] == seg)[0][0]
            on = name[idx]
        else:
            on = segname
        on = re.sub(r'[ ?*\\/|:<>"]', '_', on)

        if (vdata['data']['exportobj']['skipmodelgeneration'] == 0) and (vp.size > 0):
            base_name = f"{param['targetfileprefix']}_{seg:04d}_{on}"
            filename = os.path.join(param['targetfolder'], f"{base_name}.obj")
            
            objectname = f"{param['targetfileprefix']}_{seg:04d}_{segname}"
            mtlfilename = f"{base_name}.mtl"
            mtlfilenamewithpath = filename[:-3] + "mtl"
            materialname = f"{param['targetfileprefix']}_{seg:04d}_material"

            print(f"Exporting Surfaces ... Saving {filename} as Wavefront OBJ...")
            time.sleep(0.01)

            # Call the OBJ writing function based on Z-inversion
            if vdata['data']['exportobj']['invertz'] == 1:
                vertface2obj_mtllink(vp, fp, filename, objectname, mtlfilename, materialname)
            else:
                vertface2obj_mtllink_invnormal(vp, fp, filename, objectname, mtlfilename, materialname)

            # Determine Material Color
            if param['extractwhich'] == 10:
                # Bitwise extraction for color
                col = [
                    (seg >> 16) & 255,
                    (seg >> 8) & 255,
                    seg & 255
                ]
                col = [c / 255.0 for c in col]
            else:
                # MATLAB: colors(seg, :) -> Python 0-indexed colors[seg-1]
                # Assumes 'colors' was calculated earlier in the script
                col = colors[seg - 1, :] / 255.0

            savematerialfile(mtlfilenamewithpath, materialname, col, 1.0, 0)

        if vdata['data']['exportobj']['savesurfacestats'] == 1:
            # Update UI/Console
            print(f"Exporting Surfaces ... Evaluating surface area of {segname} ...")
            time.sleep(0.01)

            if vp.size > 0 and fp.size > 0:
                # MATLAB: tnr = segnr;
                # Use segnr as the index (ensure it matches the 0-indexed structure of param['objectsurfacearea'])
                tnr = segnr 

                # 1. Map face indices to vertex coordinates
                # MATLAB: v0=vp(fp(tri,1),:); -> Python: vp[fp[:, 0]]
                # We subtract 1 from fp if it contains 1-based MATLAB indices
                f_idx = fp.astype(int) - 1 if np.min(fp) > 0 else fp.astype(int)
                
                v0 = vp[f_idx[:, 0]]
                v1 = vp[f_idx[:, 1]]
                v2 = vp[f_idx[:, 2]]

                # 2. Vectorized Area Calculation
                # Cross product of edges (v1-v0) and (v2-v0) for all triangles at once
                # a = cross(v1-v0, v2-v0)
                a = np.cross(v1 - v0, v2 - v0)

                # 3. Magnitude and Summation
                # Area = 0.5 * sum of magnitudes of the cross products
                # MATLAB: sqrt(sum(a.*a))/2
                # np.linalg.norm computes the magnitude of the vectors
                tri_areas = np.linalg.norm(a, axis=1) / 2.0
                
                # Accumulate the total area for the current object
                param['objectsurfacearea'][tnr] += np.sum(tri_areas)
        
        segnr += 1
    
    if vdata['data']['exportobj']['savesurfacestats'] == 1:
        # Construct the full file path
        filepath = os.path.join(param['targetfolder'], vdata['data']['exportobj']['surfacestatsfile'])
        
        try:
            with open(filepath, 'w') as fid:
                # Header Information
                # MATLAB: get(vdata.fh, 'name') -> Python: Access UI title or property
                ui_name = vdata.get('fh_name', "Unknown UI") 
                fid.write(f"%% VastTools Surface Area Export\n")
                fid.write(f"%% Provided as-is, no guarantee for correctness!\n")
                fid.write(f"%% {ui_name}\n\n")
                
                fid.write(f"%% Source File: {seglayername}\n")
                fid.write(f"%% Mode: {vdata['data']['exportobj']['exportmodestring']}\n")
                
                # Area coordinates
                rp = rparam
                fid.write(f"%% Area: ({rp['xmin']}-{rp['xmax']}, {rp['ymin']}-{rp['ymax']}, {rp['zmin']}-{rp['zmax']})\n")
                
                # Voxel size calculation
                vx = param['xscale'] * param['xunit'] * mipfactx
                vy = param['yscale'] * param['yunit'] * mipfacty
                vz = param['zscale'] * param['zunit'] * vdata['data']['exportobj']['slicestep'] * mipfactz
                fid.write(f"%% Computed at voxel size: ({vx:f},{vy:f},{vz:f})\n")
                fid.write(f"%% Columns are: Object Name, Object ID, Surface Area in Export\n\n")
                
                # Loop through objects and write data
                # MATLAB: size(param.objects, 1) -> Python: len(param['objects'])
                for segnr in range(len(param['objects'])):
                    seg = int(param['objects'][segnr, 0])
                    area = param['objectsurfacearea'][segnr]
                    
                    if param['extractwhich'] == 10:
                        # Bitwise color string: col_RRGGBB
                        r = (seg >> 16) & 255
                        g = (seg >> 8) & 255
                        b = seg & 255
                        segname = f"col_{r:02X}{g:02X}{b:02X}"
                        fid.write(f'"{segname}"  {seg}  {area:f}\n')
                    else:
                        # MATLAB: name{seg} -> Python: name[seg-1] (assuming 1-based IDs)
                        obj_name = name[seg - 1]
                        fid.write(f'"{obj_name}"  {seg}  {area:f}\n')
                
                fid.write("\n")
                # File is automatically closed by the 'with' block
                
        except IOError as e:
            print(f"Error: Could not open or write to file {filepath}. {e}")


def mergemeshes(f1: np.ndarray, v1: np.ndarray, f2: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merges meshes defined by (f1, v1) and (f2, v2).
    f* are face indices (must be 1-based indices if coming from MATLAB code),
    v* are vertex coordinates.
    """
    
    # Handle empty input meshes
    if v1.size == 0:
        return f2, v2
    
    if v2.size == 0:
        return f1, v1

    nrofvertices1 = v1.shape
    
    # Offset indices in f2 to prepare for concatenation (f2 must be 1-based here)
    f2 = f2 + nrofvertices1

    # Find overlapping vertex region bounding box
    minv1 = np.min(v1, axis=)
    maxv1 = np.max(v1, axis=)
    minv2 = np.min(v2, axis=)
    maxv2 = np.max(v2, axis=)
    
    ovmin = np.maximum(minv1, minv2)
    ovmax = np.minimum(maxv1, maxv2)
    

    ov1_mask = (v1[:, 0] >= ovmin[0]) & (v1[:, 0] <= ovmax[0]) & \
               (v1[:, 1] >= ovmin[1]) & (v1[:, 1] <= ovmax[1]) & \
               (v1[:, 2] >= ovmin[2]) & (v1[:, 2] <= ovmax[2])

    ov1 = np.hstack([np.arange(1, v1.shape + 1)[:, None], v1])[ov1_mask, :]

    ov2_mask = (v2[:, 0] >= ovmin[0]) & (v2[:, 0] <= ovmax[0]) & \
               (v2[:, 1] >= ovmin[1]) & (v2[:, 1] <= ovmax[1]) & \
               (v2[:, 2] >= ovmin[2]) & (v2[:, 2] <= ovmax[2])
    ov2 = np.hstack([np.arange(1, v2.shape + 1)[:, None], v2])[ov2_mask, :]


    if ov2.size == 0:
        return np.vstack([f1, f2]), np.vstack([v1, v2])

    
    v1_coords = ov1[:, 1:]
    v2_coords = ov2[:, 1:]
    
    # Find indices of matching coordinates
    
    
    
    delete_vertex = np.zeros(v2.shape, dtype=bool)
    v2_map = np.arange(1, v2.shape + 1) # 1-based original indices of V2

    # Create a dict that maps (x,y,z) tuple back to the V1 index
    v1_map_coords = {tuple(coord): int(idx) for idx, coord in enumerate(v1, 1)}
    
    for i, coord in enumerate(v2):
        if tuple(coord) in v1_map_coords:
            v1_id = v1_map_coords[tuple(coord)]
            v2_id = v2_map[i]
            
            
            target_f2_value = v2_id + nrofvertices1
            f2[f2 == target_f2_value] = v1_id
            delete_vertex[i] = True # Mark the original v2 index for deletion

   

    # Get the indices we are keeping from v2
    kept_indices_v2 = np.where(~delete_vertex)[0]
    # The new indices for these kept vertices starts after V1
    nrofvertices1 = v1.shape
    new_indices_map = np.arange(nrofvertices1 + 1, nrofvertices1 + len(kept_indices_v2) + 1)
    
    
    
    # z = [1...nrofvertices1, 0, ..., 0] (of size nrofvertices1 + nrofvertices2)
    z = np.arange(1, nrofvertices1 + nrofvertices2 + 1) # Full 1-based ID range
    
    
    v2d = v2[~delete_vertex, :]
    
    
    
    z_map = np.zeros(nrofvertices1 + nrofvertices2 + 1, dtype=int)
    z_map[1:nrofvertices1+1] = np.arange(1, nrofvertices1 + 1) # V1 IDs map to themselves
    
   

    # Use a function to relabel faces:
    def relabel_faces(faces, deleted_mask, v1_count):
        # Create a mapping from old global indices to new global indices
        old_indices_v2 = np.arange(v1_count + 1, v1_count + len(deleted_mask) + 1)
        # New indices: 
        #   First V1_count indices are fixed
        #   Then the kept V2 indices get sequential new IDs
        
        # This is very messy. The easiest way is to use a sparse mapping or iteration here.
        
        # Let's trust the MATLAB algorithm structure is sound and use a dict map
        
        # Map original V1 indices (1 to N1) to final indices (1 to N1)
        # Map original V2 indices (N1+1 to N1+N2) to final indices (N1+1 to N1+N2-DeletedCount)
        
        new_v2_ids = np.zeros(len(delete_vertex), dtype=int)
        current_new_id = nrofvertices1 + 1
        for i, is_deleted in enumerate(delete_vertex):
            if not is_deleted:
                new_v2_ids[i] = current_new_id
                current_new_id += 1
                
        # Now use this mapping on f2 where values > nrofvertices1
        
        f2d = f2.copy()
        
        # For every face index in f2d, if it's pointing to a V2 vertex, update it
        is_v2 = f2d > nrofvertices1
        
        # Map the original V2 index (adjusted for 1-based, 0-based mismatch)
        # f2d contains indices like N1+5. We need index 5 from new_v2_ids map (index 4 in 0-based python)
        
        original_v2_idx_1based = f2d[is_v2] - nrofvertices1
        original_v2_idx_0based = original_v2_idx_1based - 1

        f2d[is_v2] = new_v2_ids[original_v2_idx_0based]

        # Return the updated faces
        return f2d
    
    f2d = relabel_faces(f2, delete_vertex, nrofvertices1)

    # Combine the final results
    f = np.vstack([f1, f2d])
    v = np.vstack([v1, v2d])
    
    return f, v

def vertface2obj_mtllink_invnormal(v, f, filename, objectname, mtlfilename, materialname):
    """
    Saves mesh as OBJ with inverted normals (flipped winding order).
    """
    with open(filename, 'w') as fid:
        fid.write(f"mtllib {mtlfilename}\n")
        fid.write(f"usemtl {materialname}\n")
        
        # Write vertices
        for row in v:
            fid.write(f"v {row[0]:f} {row[1]:f} {row[2]:f}\n")
            
        fid.write(f"g {objectname}\n")
        
        # Write faces with flipped order: f(i,2), f(i,1), f(i,3)
        # Note: assuming f contains 1-based indices
        for row in f:
            fid.write(f"f {int(row[1])} {int(row[0])} {int(row[2])}\n")
            
        fid.write("g\n")

def vertface2obj_mtllink(v, f, filename, objectname, mtlfilename, materialname):
    """
    Saves mesh as OBJ with standard winding order.
    """
    with open(filename, 'w') as fid:
        # Header
        fid.write(f"mtllib {mtlfilename}\n")
        fid.write(f"usemtl {materialname}\n")
        
        # Write all vertices efficiently
        # Creates a single large string block for all vertices
        v_lines = "".join(f"v {row[0]:f} {row[1]:f} {row[2]:f}\n" for row in v)
        fid.write(v_lines)
        
        fid.write(f"g {objectname}\n")
        
        # Write all faces efficiently
        # Note: assuming f contains 1-based indices
        f_lines = "".join(f"f {int(row[0])} {int(row[1])} {int(row[2])}\n" for row in f)
        fid.write(f_lines)
        
        fid.write("g\n")


"""
def export_segment_meshes(miplevel=0, output_file=None):

    vast = VASTControlClass()
    if not vast.connect():
        print("Failed to connect")
        return
    
    info = vast.get_info()

    # get number of segments
    nr_segments = vast.get_number_of_segments()
    if nr_segments == 0:  
        print("ERROR: Cannot export models from segments. No segmentation available in VAST.")
        return

    extract_seg = True  

    # Get all segment data
    all_seg_data = vast.get_all_segment_data()  # Returns list of dicts
    names = vast.get_all_segment_names()  # Returns list of strings

    # Get selected segmentation layer
    selected_layers = vast.get_selected_layer_nr()
    if not selected_layers or selected_layers.get('selected_segment_layer', -1) < 0:
        print("No segment layer selected")
        vast.disconnect()
        return

    seg_layer_nr = selected_layers['selected_segment_layer']

    # Get layer info to get the name
    seg_layer_info = vast.get_layer_info(seg_layer_nr)
    if seg_layer_info:
        seg_layer_name = seg_layer_info['name']
    else:
        seg_layer_name = f"Layer_{seg_layer_nr}"

    # Remove background name (first entry)
    if names and len(names) > 0:
        names = names[1:]  # Remove first element (Background)

    # Get max object number
    max_object_number = max([seg['id'] for seg in all_seg_data]) if all_seg_data else 0

    # Get mip data size
    mip_data_size = vast.get_data_size_at_mip(seg_layer_nr, miplevel)



    # seg_data = vast.get_segment_data(segment_id)
    # if not seg_data:
    #     return None
    
    # bbox = seg_data['boundingbox']
    # minx, miny, minz, maxx, maxy, maxz = bbox
    
    # print(f"Fetching segment {segment_id}...")
    
    # voxels = vast.get_seg_image_rle_decoded(
    #     miplevel=miplevel,
    #     minx=minx, maxx=maxx,
    #     miny=miny, maxy=maxy,
    #     minz=minz, maxz=maxz
    # )
    
    # if voxels is None:
    #     return None
    
    # mask = (voxels == segment_id).astype(np.uint8)
    
    # print("Running marching cubes...")
    # verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    
    # # Scale to nm
    # scale = np.array([info['voxelsizex'], info['voxelsizey'], info['voxelsizez']])
    # offset = np.array([minx, miny, minz])
    
    # verts_scaled = verts * scale + offset * scale
    
    # if output_file:
    #     if output_file.endswith('.obj'):
    #         save_obj(output_file, verts_scaled, faces)
    #     elif output_file.endswith('.stl'):
    #         save_stl(output_file, verts_scaled, faces)
    #     print(f"Saved {output_file}")
    
    # return verts_scaled, faces
    vast.disconnect()
"""

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


def save_material_file(filename, material_name, color, alpha, roughness=0.5):
    """Save MTL (material) file for OBJ."""
    with open(filename, 'w') as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # Ambient
        f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # Diffuse
        f.write(f"Ks 0.5 0.5 0.5\n")  # Specular
        f.write(f"d {alpha:.3f}\n")  # Transparency
        f.write(f"Ns 32\n")  # Shininess


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
    