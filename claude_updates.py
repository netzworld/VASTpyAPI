import numpy as np
from skimage import measure
from scipy import ndimage
from math import floor
import re
from typing import Tuple, List, Dict, Optional
import time
from VASTControlClass import VASTControlClass


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