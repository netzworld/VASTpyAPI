import numpy as np
from skimage import measure
from scipy import ndimage
from math import floor, ceil
import re
from typing import Tuple, List, Dict, Optional
import time
import logging
from VASTControlClass import VASTControlClass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SurfaceExtractor:
    """
    Python port of MATLAB extractsurfaces function.
    Extracts 3D surfaces from segmentation or screenshot data.
    """
    
    def __init__(self, vast_control: VASTControlClass, export_params: Dict, region_params: Dict):
        """
        Initialize the surface extractor.
        
        Args:
            vast_control: VASTControlClass instance for data access
            export_params: Dictionary containing export configuration
            region_params: Dictionary containing region bounds (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        self.vast = vast_control
        self.export_params = export_params
        self.region_params = region_params
        self.canceled = False
        
        # Will be populated during extraction
        self.param = {}
        self.objects = None
        self.names = []
        self.data = None
        
    def extract_surfaces(self):
        """Main extraction function - Part 1: Setup and initialization"""
        
        # Check connection (implement as needed for your GUI)
        if not self._check_connection():
            return
        
        # Disable network warnings if requested
        if self.export_params.get('disablenetwarnings', 0) == 1:
            self.vast.set_error_popups_enabled(74, False)  # network connection error
            self.vast.set_error_popups_enabled(75, False)  # unexpected data
            self.vast.set_error_popups_enabled(76, False)  # unexpected length
        
        self._update_message('Exporting Surfaces ...', 'Loading Metadata ...')
        
        param = self.export_params.copy()
        rparam = self.region_params.copy()
        
        # Determine if extracting segmentation or screenshots
        extract_seg = True
        if 5 <= param.get('extractwhich', 1) <= 10:
            extract_seg = False
        
        # Load segmentation data or setup screenshot extraction
        if extract_seg:
            # Get all segment data
            data, res = self.vast.get_all_segment_data_matrix()
            names, res = self.vast.get_all_segment_names()
            seg_layer_name = self._get_selected_seg_layer_name()
            
            # Remove 'Background' (first entry)
            names = names[1:]
            
            max_object_number = int(np.max(data[:, 0]))
            
            # Get selected layer info
            selected_layer_nr, selected_em_layer_nr, selected_seg_layer_nr, res = \
                self.vast.get_selected_layer_nr()
            
            mip_data_size = self.vast.get_data_size_at_mip(
                param['miplevel'], 
                selected_seg_layer_nr
            )
        else:
            # Setup for screenshot extraction
            names = self._setup_screenshot_names(param)
            selected_layer_nr, selected_em_layer_nr, selected_seg_layer_nr, res = \
                self.vast.get_selected_layer_nr()
        
        # Get mipmap scale factors
        z_scale = 1
        z_min = rparam['zmin']
        z_max = rparam['zmax']
        
        if param['miplevel'] > 0:
            if extract_seg:
                mip_scale_matrix, res = self.vast.get_mipmap_scale_factors(selected_seg_layer_nr)
            else:
                mip_scale_matrix, res = self.vast.get_mipmap_scale_factors(selected_em_layer_nr)
            
            z_scale = mip_scale_matrix[param['miplevel'], 2]
            
            if z_scale != 1:
                z_min = floor(z_min / z_scale)
                z_max = floor(z_max / z_scale)
        
        # Calculate coordinate bounds at current mip level
        x_min = rparam['xmin'] >> param['miplevel']
        x_max = (rparam['xmax'] >> param['miplevel']) - 1
        y_min = rparam['ymin'] >> param['miplevel']
        y_max = (rparam['ymax'] >> param['miplevel']) - 1
        
        # Get mip scale factors
        if param['miplevel'] > 0:
            mip_fact_x = mip_scale_matrix[param['miplevel'], 0]
            mip_fact_y = mip_scale_matrix[param['miplevel'], 1]
            mip_fact_z = mip_scale_matrix[param['miplevel'], 2]
        else:
            mip_fact_x = 1
            mip_fact_y = 1
            mip_fact_z = 1
        
        # Validate volume dimensions
        if ((x_min == x_max or y_min == y_max or z_min == z_max) and 
            param.get('closesurfaces', 0) == 0):
            self._show_error(
                'ERROR: The surface script needs a volume which is at least '
                'two pixels wide in each direction. Please adjust "Render from area" '
                'values, or enable "Close surface sides".'
            )
            self._update_message('Canceled.')
            return
        
        # Store processed parameters
        self.param = {
            'extract_seg': extract_seg,
            'names': names,
            'xmin': x_min,
            'xmax': x_max,
            'ymin': y_min,
            'ymax': y_max,
            'zmin': z_min,
            'zmax': z_max,
            'mipfactx': mip_fact_x,
            'mipfacty': mip_fact_y,
            'mipfactz': mip_fact_z,
            **param
        }
        
        if extract_seg:
            self.data = data
            self.param['max_object_number'] = max_object_number
            self.param['mip_data_size'] = mip_data_size
            self.param['seg_layer_name'] = seg_layer_name
        
        print(f"Initialization complete. Extract mode: {'segmentation' if extract_seg else 'screenshots'}")
        
        # Continue to Part 2
        self._process_objects_and_blocks(extract_seg, data, names, mip_scale_matrix)
        
    def _setup_screenshot_names(self, param: Dict) -> List[str]:
        """Setup names for screenshot extraction modes"""
        extract_which = param.get('extractwhich', 5)
        
        if extract_which == 5:  # RGB 50%
            return ['Red Layer', 'Green Layer', 'Blue Layer']
        elif extract_which == 6:  # Brightness 50%
            param['lev'] = 128
            return ['Brightness 128']
        elif extract_which == 7:  # 16 levels
            param['lev'] = list(range(8, 257, 16))
            return [f'B{lev:03d}' for lev in param['lev']]
        elif extract_which == 8:  # 32 levels
            param['lev'] = list(range(4, 257, 8))
            return [f'B{lev:03d}' for lev in param['lev']]
        elif extract_which == 9:  # 64 levels
            param['lev'] = list(range(2, 257, 4))
            return [f'B{lev:03d}' for lev in param['lev']]
        else:
            return []
    
    def _check_connection(self) -> bool:
        """Check if connection to VAST is valid"""
        # Implement based on your GUI/connection requirements
        return True
    
    def _get_selected_seg_layer_name(self) -> str:
        """Get the name of the selected segmentation layer"""
        # Implement via VAST API
        return self.vast.get_selected_seg_layer_name()
    
    def _update_message(self, *messages):
        """Update progress message (implement for your GUI)"""
        print(' | '.join(messages))
        time.sleep(0.01)
    
    def _show_error(self, message: str):
        """Show error message (implement for your GUI)"""
        print(f"ERROR: {message}")
    
    def _process_objects_and_blocks(self, extract_seg: bool, data: np.ndarray, 
                                     names: List[str], mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 2: Process object selection, compute folder names, and calculate block divisions
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
            data: Segment data matrix (for segmentation mode)
            names: List of object/segment names
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        
        if extract_seg:
            # ===== COMPUTE FULL NAMES (including folder hierarchy) =====
            if param.get('includefoldernames', 0) == 1:
                logging.info("Computing full names with folder hierarchy...")
                full_names = names.copy()
                for i in range(len(data)):
                    j = i
                    # Traverse up the hierarchy (column 14 is parent ID)
                    while data[j, 13] != 0:  # Check if parent is not 0
                        j = int(data[j, 13])
                        full_names[i] = names[j] + '.' + full_names[i]
                names = full_names
                logging.debug(f"Sample full name: {names[0] if names else 'N/A'}")
            
            # ===== COMPUTE LIST OF OBJECTS TO EXPORT =====
            extract_which = param.get('extractwhich', 1)
            logging.info(f"Computing object list (extractwhich={extract_which})...")
            
            if extract_which == 1:
                # All segments individually, uncollapsed
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                self.vast.set_seg_translation(None, None)
                logging.info(f"Mode 1: Exporting {len(objects)} individual segments")
                
            elif extract_which == 2:
                # All segments, collapsed as in VAST
                objects = np.unique(data[:, 17])  # Column 18 in MATLAB (0-indexed: 17)
                objects = np.column_stack([objects, data[objects.astype(int), 1]]).astype(np.uint32)
                self.vast.set_seg_translation(data[:, 0], data[:, 17])
                logging.info(f"Mode 2: Exporting {len(objects)} collapsed segments")
                
            elif extract_which == 3:
                # Selected segment and children, uncollapsed
                selected = np.where((data[:, 1] & 65536) > 0)[0]
                if len(selected) == 0:
                    objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
                    logging.warning("No segments selected, exporting all")
                else:
                    selected = np.concatenate([selected, self._get_child_tree_ids_seg(data, selected)])
                    objects = np.column_stack([selected, data[selected, 1]]).astype(np.uint32)
                    logging.info(f"Mode 3: Exporting {len(objects)} selected segments + children")
                self.vast.set_seg_translation(data[selected, 0], data[selected, 0])
                
            elif extract_which == 4:
                # Selected segment and children, collapsed as in VAST
                selected = np.where((data[:, 1] & 65536) > 0)[0]
                if len(selected) == 0:
                    # None selected: choose all, collapsed
                    selected = data[:, 0].astype(int)
                    objects = np.unique(data[:, 17])
                    logging.warning("No segments selected, exporting all collapsed")
                else:
                    selected = np.concatenate([selected, self._get_child_tree_ids_seg(data, selected)])
                    objects = np.unique(data[selected.astype(int), 17])
                    logging.info(f"Mode 4: Exporting {len(objects)} collapsed selected segments + children")
                
                objects = np.column_stack([objects, data[objects.astype(int), 1]]).astype(np.uint32)
                self.vast.set_seg_translation(data[selected.astype(int), 0], data[selected.astype(int), 17])
            
            else:
                logging.error(f"Unknown extractwhich value: {extract_which}")
                objects = np.column_stack([data[:, 0], data[:, 1]]).astype(np.uint32)
            
            self.objects = objects
            param['objects'] = objects
        
        # ===== COMPUTE NUMBER OF BLOCKS/TILES IN VOLUME =====
        logging.info("Computing block divisions...")
        
        x_min, x_max = param['xmin'], param['xmax']
        y_min, y_max = param['ymin'], param['ymax']
        z_min, z_max = param['zmin'], param['zmax']
        
        block_size_x = param.get('blocksizex', 256)
        block_size_y = param.get('blocksizey', 256)
        block_size_z = param.get('blocksizez', 256)
        overlap = param.get('overlap', 0)
        slice_step = param.get('slicestep', 1)
        
        # Calculate X tiles
        nr_x_tiles = 0
        tile_x1 = x_min
        while tile_x1 <= x_max:
            tile_x1 += block_size_x - overlap
            nr_x_tiles += 1
        
        # Calculate Y tiles
        nr_y_tiles = 0
        tile_y1 = y_min
        while tile_y1 <= y_max:
            tile_y1 += block_size_y - overlap
            nr_y_tiles += 1
        
        # Calculate Z tiles
        nr_z_tiles = 0
        tile_z1 = z_min
        
        if slice_step == 1:
            slice_numbers = list(range(z_min, z_max + 1))
            while tile_z1 <= z_max:
                tile_z1 += block_size_z - overlap
                nr_z_tiles += 1
        else:
            slice_numbers = list(range(z_min, z_max + 1, slice_step))
            nr_z_tiles = ceil(len(slice_numbers) / (block_size_z - overlap))
            
            # Compute slice numbers for each block
            block_slice_numbers = []
            j = 0
            for p in range(0, len(slice_numbers), block_size_z - overlap):
                pe = min(p + block_size_z, len(slice_numbers))
                block_slice_numbers.append(slice_numbers[p:pe])
                j += 1
            param['block_slice_numbers'] = block_slice_numbers
        
        param['nr_x_tiles'] = nr_x_tiles
        param['nr_y_tiles'] = nr_y_tiles
        param['nr_z_tiles'] = nr_z_tiles
        param['slice_numbers'] = slice_numbers
        
        print(f"Block division: {nr_x_tiles} x {nr_y_tiles} x {nr_z_tiles} = {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")
        logging.info(f"Total blocks to process: {nr_x_tiles * nr_y_tiles * nr_z_tiles}")
        
        # Continue to Part 3 (MIP region constraint)
        if param.get('usemipregionconstraint', 0) == 1:
            self._compute_mip_region_constraint(extract_seg, mip_scale_matrix)
    
    def _get_child_tree_ids_seg(self, data: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
        """
        Recursively get all child IDs in the segment hierarchy
        
        Args:
            data: Segment data matrix
            parent_ids: Array of parent segment indices
            
        Returns:
            Array of all child indices
        """
        children = []
        for pid in parent_ids:
            # Find all segments where parent (column 14/index 13) equals this segment's ID
            child_mask = data[:, 13] == data[pid, 0]
            child_indices = np.where(child_mask)[0]
            if len(child_indices) > 0:
                children.extend(child_indices)
                # Recursively get children of children
                children.extend(self._get_child_tree_ids_seg(data, child_indices))
        
        return np.array(children, dtype=int) if children else np.array([], dtype=int)
    
    def _compute_mip_region_constraint(self, extract_seg: bool, mip_scale_matrix: Optional[np.ndarray]):
        """
        Part 3: Compute MIP region constraint to skip empty blocks
        
        This loads a lower-resolution version of the data, creates a binary mask,
        dilates it by padding, and marks which blocks contain data.
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
            mip_scale_matrix: Mipmap scaling factors
        """
        param = self.param
        rparam = {k: param[k] for k in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] if k in param}
        
        logging.info("Computing MIP region constraint...")
        print("Loading constraint mask at lower resolution...")
        
        # Calculate export region at constraint mip level
        mip_region_mip = param.get('mipregionmip', 0)
        
        c_x_min = rparam['xmin'] >> mip_region_mip
        c_x_max = (rparam['xmax'] >> mip_region_mip) - 1
        c_y_min = rparam['ymin'] >> mip_region_mip
        c_y_max = (rparam['ymax'] >> mip_region_mip) - 1
        c_z_min = rparam['zmin']
        c_z_max = rparam['zmax']
        
        if mip_region_mip > 0 and mip_scale_matrix is not None:
            c_z_scale = mip_scale_matrix[mip_region_mip][2]
            c_z_min = floor(c_z_min / c_z_scale)
            c_z_max = floor(c_z_max / c_z_scale)
        
        # Load complete region at constraint mip level
        if extract_seg:
            # Load segmentation data
            print(f"Loading segmentation at mip {mip_region_mip} for constraint...")
            logging.info(f"Loading segmentation constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
            
            slice_step = param.get('slicestep', 1)
            
            if slice_step == 1:
                mc_seg_image, values, numbers, bboxes, res = \
                    self.vast.get_seg_image_rle_decoded_bboxes(
                        mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, 0
                    )
            else:
                # Load slice by slice for non-unit step
                slices = list(range(c_z_min, c_z_max + 1, slice_step))
                mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                
                for idx, slice_num in enumerate(slices):
                    mc_seg_slice, values, numbers, bboxes, res = \
                        self.vast.get_seg_image_rle_decoded_bboxes(
                            mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                            slice_num, slice_num, 0
                        )
                    mc_seg_image[:, :, idx] = mc_seg_slice
        else:
            # Load screenshot data
            print(f"Loading screenshots at mip {mip_region_mip} for constraint...")
            logging.info(f"Loading screenshot constraint: ({c_x_min}-{c_x_max}, {c_y_min}-{c_y_max}, {c_z_min}-{c_z_max})")
            
            slice_step = param.get('slicestep', 1)
            
            if slice_step == 1:
                mc_seg_image, res = self.vast.get_screenshot_image(
                    mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, c_z_min, c_z_max, 0
                )
                # Convert from (y, x, z, c) to (x, y, z) and make binary
                mc_seg_image = np.transpose(np.sum(mc_seg_image, axis=3) > 0, (1, 0, 2))
            else:
                slices = list(range(c_z_min, c_z_max + 1, slice_step))
                mc_seg_image = np.zeros((c_x_max - c_x_min + 1, c_y_max - c_y_min + 1, len(slices)))
                
                for idx, slice_num in enumerate(slices):
                    mc_seg_slice, res = self.vast.get_screenshot_image(
                        mip_region_mip, c_x_min, c_x_max, c_y_min, c_y_max, 
                        slice_num, slice_num, 0
                    )
                    mc_seg_image[:, :, idx] = mc_seg_slice
                
                mc_seg_image = np.transpose(np.sum(mc_seg_image, axis=3) > 0, (1, 0, 2))
        
        # Convert to binary mask (VAST pre-translates so nonzero = exported objects)
        print("Processing constraint mask...")
        mc_seg_image = (mc_seg_image > 0).astype(np.uint8)
        
        # Dilate mask by region padding
        padding = param.get('mipregionpadding', 1)
        if padding > 0:
            kernel_size = padding * 2 + 1
            strel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
            mc_seg_image = ndimage.binary_dilation(mc_seg_image, structure=strel).astype(np.uint8)
            logging.info(f"Dilated mask by {padding} pixels")
        
        # Generate 3D matrix of block loading flags
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']
        mc_load_flags = np.zeros((nr_x_tiles, nr_y_tiles, nr_z_tiles), dtype=bool)
        
        # Calculate scale factors between constraint mip and export mip
        if mip_scale_matrix is not None:
            c_mip_fact_x = mip_scale_matrix[mip_region_mip][0] / mip_scale_matrix[param['miplevel']][0]
            c_mip_fact_y = mip_scale_matrix[mip_region_mip][1] / mip_scale_matrix[param['miplevel']][1]
            c_mip_fact_z = mip_scale_matrix[mip_region_mip][2] / mip_scale_matrix[param['miplevel']][2]
        else:
            c_mip_fact_x = c_mip_fact_y = c_mip_fact_z = 1
        
        # Iterate through all blocks and check if they contain data
        x_min, y_min, z_min = param['xmin'], param['ymin'], param['zmin']
        x_max, y_max, z_max = param['xmax'], param['ymax'], param['zmax']
        block_size_x = param.get('blocksizex', 256)
        block_size_y = param.get('blocksizey', 256)
        block_size_z = param.get('blocksizez', 256)
        overlap = param.get('overlap', 0)
        
        print("Checking blocks for data...")
        tile_z1 = z_min
        for tz in range(nr_z_tiles):
            tile_z2 = min(tile_z1 + block_size_z - 1, z_max)
            tile_y1 = y_min
            
            for ty in range(nr_y_tiles):
                tile_y2 = min(tile_y1 + block_size_y - 1, y_max)
                tile_x1 = x_min
                
                for tx in range(nr_x_tiles):
                    tile_x2 = min(tile_x1 + block_size_x - 1, x_max)
                    
                    # Compute tile coords on constraint mip
                    c_min_x = max(0, int(floor((tile_x1 - x_min) / c_mip_fact_x)))
                    c_max_x = min(mc_seg_image.shape[0] - 1, int(ceil((tile_x2 - x_min) / c_mip_fact_x)))
                    c_min_y = max(0, int(floor((tile_y1 - y_min) / c_mip_fact_y)))
                    c_max_y = min(mc_seg_image.shape[1] - 1, int(ceil((tile_y2 - y_min) / c_mip_fact_y)))
                    c_min_z = max(0, int(floor((tile_z1 - z_min) / c_mip_fact_z)))
                    c_max_z = min(mc_seg_image.shape[2] - 1, int(ceil((tile_z2 - z_min) / c_mip_fact_z)))
                    
                    # Crop region from constraint mip
                    crop_region = mc_seg_image[c_min_x:c_max_x+1, c_min_y:c_max_y+1, c_min_z:c_max_z+1]
                    
                    # Flag will be True if any voxels in crop region are nonzero
                    mc_load_flags[tx, ty, tz] = np.max(crop_region) > 0
                    
                    tile_x1 += block_size_x - overlap
                tile_y1 += block_size_y - overlap
            tile_z1 += block_size_z - overlap
        
        blocks_to_process = np.sum(mc_load_flags)
        blocks_to_skip = mc_load_flags.size - blocks_to_process
        print(f"Constraint check complete: {blocks_to_process} blocks to process, {blocks_to_skip} blocks to skip")
        logging.info(f"MIP constraint reduces workload by {100 * blocks_to_skip / mc_load_flags.size:.1f}%")
        
        param['mc_load_flags'] = mc_load_flags
    
    def _initialize_storage_arrays(self, extract_seg: bool):
        """
        Part 4: Initialize storage arrays for faces and vertices
        
        Creates cell/dict structures to store mesh data for each object in each block.
        For mode 10 (unique colors), uses sparse indexing.
        
        Args:
            extract_seg: True if extracting segmentation, False for screenshots
        """
        param = self.param
        
        logging.info("Initializing storage arrays...")
        
        nr_x_tiles = param['nr_x_tiles']
        nr_y_tiles = param['nr_y_tiles']
        nr_z_tiles = param['nr_z_tiles']
        
        if extract_seg:
            # Storage for segmentation objects
            max_obj_num = param['max_object_number']
            
            # Initialize as nested dictionaries: farray[obj][tx][ty][tz] = faces
            # Using dict for sparse storage (most cells will be empty)
            param['farray'] = {}
            param['varray'] = {}
            param['object_volume'] = np.zeros(len(param['objects']), dtype=np.int64)
            
            print(f"Initialized storage for {max_obj_num} potential objects across {nr_x_tiles}x{nr_y_tiles}x{nr_z_tiles} blocks")
            logging.info(f"Storage size: {len(param['objects'])} objects x {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")
            
        else:
            # Storage for screenshot-based extraction
            extract_which = param.get('extractwhich', 5)
            
            if extract_which == 10:
                # Up to 2^24 unique color objects
                # Use sparse matrix for indexing: fvindex[color, block_idx] = block_number
                total_colors = 256 * 256 * 256
                total_blocks = (nr_x_tiles + 1) * (nr_y_tiles + 1) * (nr_z_tiles + 1)
                
                # Sparse matrix would be huge, use dict instead
                param['fvindex'] = {}  # Will map (color, block_idx) -> block_number
                param['farray'] = {}   # Will map block_number -> faces
                param['varray'] = {}   # Will map block_number -> vertices
                param['object_volume'] = np.zeros(total_colors, dtype=np.int64)
                
                print(f"Initialized storage for unique color extraction (up to {total_colors} colors)")
                logging.info("Using sparse storage for color-based extraction")
                
            else:
                # RGB layers or brightness levels (small number of objects)
                num_objects = len(param['objects'])
                
                param['farray'] = {}
                param['varray'] = {}
                param['object_volume'] = np.zeros(num_objects, dtype=np.int64)
                
                print(f"Initialized storage for {num_objects} screenshot-based objects")
                logging.info(f"Storage for {num_objects} objects x {nr_x_tiles * nr_y_tiles * nr_z_tiles} blocks")


# Example usage:
if __name__ == "__main__":
    # Initialize VAST control
    vast = VASTControlClass()
    
    # Setup export parameters
    export_params = {
        'extractwhich': 1,  # 1=all segments, 2=collapsed, 3=selected+children, etc.
        'miplevel': 0,
        'blocksizex': 256,
        'blocksizey': 256,
        'blocksizez': 256,
        'overlap': 0,
        'closesurfaces': 0,
        'slicestep': 1,
        'disablenetwarnings': 1,
        'xscale': 1.0,
        'yscale': 1.0,
        'zscale': 1.0,
        'xunit': 1.0,
        'yunit': 1.0,
        'zunit': 1.0,
    }
    
    # Setup region parameters
    region_params = {
        'xmin': 0,
        'xmax': 512,
        'ymin': 0,
        'ymax': 512,
        'zmin': 0,
        'zmax': 100,
    }
    
    # Create extractor and run
    extractor = SurfaceExtractor(vast, export_params, region_params)
    extractor.extract_surfaces()