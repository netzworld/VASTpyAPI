import numpy as np
from skimage import measure
from VASTControlClass import VASTControlClass

def export_segment_mesh(vast, segment_id, miplevel=0, output_file=None):
    """Export segment as triangular mesh (OBJ/STL)."""
    
    info = vast.get_info()
    if not info:
        return None
    
    seg_data = vast.get_segment_data(segment_id)
    if not seg_data:
        return None
    
    bbox = seg_data['boundingbox']
    minx, miny, minz, maxx, maxy, maxz = bbox
    
    print(f"Fetching segment {segment_id}...")
    
    voxels = vast.get_seg_image_rle_decoded(
        miplevel=miplevel,
        minx=minx, maxx=maxx,
        miny=miny, maxy=maxy,
        minz=minz, maxz=maxz
    )
    
    if voxels is None:
        return None
    
    mask = (voxels == segment_id).astype(np.uint8)
    
    print("Running marching cubes...")
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    
    # Scale to nm
    scale = np.array([info['voxelsizex'], info['voxelsizey'], info['voxelsizez']])
    offset = np.array([minx, miny, minz])
    
    verts_scaled = verts * scale + offset * scale
    
    if output_file:
        if output_file.endswith('.obj'):
            save_obj(output_file, verts_scaled, faces)
        elif output_file.endswith('.stl'):
            save_stl(output_file, verts_scaled, faces)
        print(f"Saved {output_file}")
    
    return verts_scaled, faces


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


# Example usage
if __name__ == "__main__":
    vast = VASTControlClass()
    
    if not vast.connect():
        print("Failed to connect")
        exit(1)
    
    # Export segment mesh
    export_segment_mesh(vast, segment_id=1, miplevel=2, output_file="segment_1.obj")
    
    # Export skeleton
    export_skeleton(vast, anno_object_id=1, output_file="skeleton_1.swc")
    
    vast.disconnect()