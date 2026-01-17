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