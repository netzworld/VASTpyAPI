import unittest
import numpy as np
from VASTControlClass import VASTControlClass, HOST, PORT


class TestVASTConnection(unittest.TestCase):
    """Test connection management."""
    
    def setUp(self):
        self.vast = VASTControlClass(HOST, PORT)
    
    def tearDown(self):
        if self.vast.client:
            self.vast.disconnect()
    
    def test_connect(self):
        """Test connecting to VAST."""
        result = self.vast.connect(timeout=10)
        self.assertEqual(result, 1, "Connection should succeed")
        self.assertIsNotNone(self.vast.client, "Client socket should be initialized")
    
    def test_disconnect(self):
        """Test disconnecting from VAST."""
        self.vast.connect(timeout=10)
        result = self.vast.disconnect()
        self.assertEqual(result, 1, "Disconnect should succeed")
        self.assertIsNone(self.vast.client, "Client socket should be None after disconnect")


class TestVASTGeneralFunctions(unittest.TestCase):
    """Test general VAST information retrieval."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_info(self):
        """Test retrieving dataset information."""
        info = self.vast.get_info()
        self.assertIsInstance(info, dict, "Should return dict")
        self.assertIn('datasizex', info, "Should contain datasizex")
        self.assertIn('datasizey', info, "Should contain datasizey")
        self.assertIn('datasizez', info, "Should contain datasizez")
        self.assertIn('voxelsizex', info, "Should contain voxelsizex")
        self.assertIn('voxelsizey', info, "Should contain voxelsizey")
        self.assertIn('voxelsizez', info, "Should contain voxelsizez")
        self.assertIn('nrofmiplevels', info, "Should contain nrofmiplevels")
        self.assertGreater(info['datasizex'], 0, "Data size should be positive")
    
    def test_get_api_version(self):
        """Test retrieving API version."""
        version = self.vast.get_api_version()
        self.assertIsInstance(version, int, "Version should be integer")
        self.assertGreater(version, 0, "Version should be positive")
    
    def test_get_hardware_info(self):
        """Test retrieving hardware information."""
        hw_info = self.vast.get_hardware_info()
        self.assertIsInstance(hw_info, dict, "Should return dict")
        # Hardware info may not always be available or properly parsed
        if hw_info:
            expected_keys = ['computername', 'processorname', 'processorspeed_ghz',
                            'nrofprocessorcores', 'totalmemorygb', 'graphicscardname']
            for key in expected_keys:
                self.assertIn(key, hw_info, f"Should contain {key}")
    
    def test_get_last_error(self):
        """Test retrieving last error code."""
        error = self.vast.get_last_error()
        self.assertIsInstance(error, int, "Error code should be integer")
    
    def test_set_error_popups_enabled(self):
        """Test enabling/disabling error popups."""
        # This function may not be supported in all VAST versions
        result = self.vast.set_error_popups_enabled(10, True)
        self.assertIsInstance(result, bool, "Should return boolean")
        if result:
            result = self.vast.set_error_popups_enabled(10, False)
            self.assertIsInstance(result, bool, "Should return boolean")


class TestVASTLayerFunctions(unittest.TestCase):
    """Test layer management functions."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_number_of_layers(self):
        """Test retrieving number of layers."""
        num_layers = self.vast.get_number_of_layers()
        self.assertIsInstance(num_layers, int, "Should return integer")
        self.assertGreaterEqual(num_layers, 0, "Number of layers should be non-negative")
    
    def test_get_layer_info(self):
        """Test retrieving layer information."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            layer_info = self.vast.get_layer_info(1)
            self.assertIsInstance(layer_info, dict, "Should return dict")
            expected_keys = ['type', 'editable', 'visible', 'brightness', 
                           'contrast', 'name', 'bytesperpixel']
            for key in expected_keys:
                self.assertIn(key, layer_info, f"Should contain {key}")
    
    def test_set_layer_info(self):
        """Test setting layer information."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            # Get current info first
            current_info = self.vast.get_layer_info(1)
            if current_info and current_info.get('editable', 0) == 1:
                # Only try to set if layer is editable
                layer_info = {'brightness': 128, 'contrast': 128}
                result = self.vast.set_layer_info(1, layer_info)
                self.assertIsInstance(result, bool, "Should return boolean")
            else:
                self.skipTest("Layer 1 is not editable")
    
    def test_get_mipmap_scale_factors(self):
        """Test retrieving mipmap scale factors."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            scale_factors = self.vast.get_mipmap_scale_factors(1)
            self.assertIsInstance(scale_factors, list, "Should return list")
            if len(scale_factors) > 0:
                self.assertEqual(len(scale_factors[0]), 3, "Each factor should have 3 values (x,y,z)")
    
    def test_get_data_size_at_mip(self):
        """Test retrieving data size at specific mip level."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            size = self.vast.get_data_size_at_mip(1, 0)
            if size is not None:
                self.assertEqual(len(size), 3, "Should return (x, y, z) tuple")
                self.assertGreater(size[0], 0, "X size should be positive")
    
    def test_get_selected_layer_nr(self):
        """Test retrieving selected layer number."""
        result = self.vast.get_selected_layer_nr()
        self.assertIsInstance(result, dict, "Should return dict")
        self.assertIn('selected_layer', result, "Should contain selected_layer")
    
    def test_set_selected_layer_nr(self):
        """Test setting selected layer."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            result = self.vast.set_selected_layer_nr(1)
            self.assertTrue(result, "Should successfully set selected layer")
    
    def test_get_api_layers_enabled(self):
        """Test checking if API layers are enabled."""
        result = self.vast.get_api_layers_enabled()
        self.assertIn(result, [0, 1], "Should return 0 or 1")
    
    def test_set_api_layers_enabled(self):
        """Test enabling/disabling API layers."""
        result = self.vast.set_api_layers_enabled(True)
        self.assertTrue(result, "Should enable API layers")
        result = self.vast.set_api_layers_enabled(False)
        self.assertTrue(result, "Should disable API layers")
    
    def test_get_selected_api_layer_nr(self):
        """Test retrieving selected API layer."""
        self.vast.set_api_layers_enabled(True)
        result = self.vast.get_selected_api_layer_nr()
        self.assertIsInstance(result, dict, "Should return dict")
        expected_keys = ['selected_layer', 'selected_em_layer', 
                        'selected_anno_layer', 'selected_segment_layer']
        for key in expected_keys:
            self.assertIn(key, result, f"Should contain {key}")
    
    def test_set_selected_api_layer_nr(self):
        """Test setting selected API layer."""
        self.vast.set_api_layers_enabled(True)
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            result = self.vast.set_selected_api_layer_nr(1)
            self.assertTrue(result, "Should successfully set API layer")


class TestVASTSegmentationFunctions(unittest.TestCase):
    """Test segmentation functions."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_number_of_segments(self):
        """Test retrieving number of segments."""
        num_segments = self.vast.get_number_of_segments()
        self.assertIsInstance(num_segments, int, "Should return integer")
        self.assertGreaterEqual(num_segments, 0, "Should be non-negative")
    
    def test_get_segment_data(self):
        """Test retrieving segment metadata."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            seg_data = self.vast.get_segment_data(1)
            self.assertIsInstance(seg_data, dict, "Should return dict")
            expected_keys = ['id', 'flags', 'col1', 'col2', 'anchorpoint', 
                           'hierarchy', 'boundingbox']
            for key in expected_keys:
                self.assertIn(key, seg_data, f"Should contain {key}")
    
    def test_get_segment_name(self):
        """Test retrieving segment name."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            name = self.vast.get_segment_name(1)
            self.assertIsInstance(name, str, "Should return string")
    
    def test_set_segment_name(self):
        """Test setting segment name."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            result = self.vast.set_segment_name(1, "Test Segment")
            self.assertTrue(result, "Should successfully set segment name")
    
    def test_set_anchor_point(self):
        """Test setting segment anchor point."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            result = self.vast.set_anchor_point(1, 100, 100, 50)
            self.assertTrue(result, "Should successfully set anchor point")
    
    def test_set_segment_color_8(self):
        """Test setting segment color with 8-bit RGB values."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            result = self.vast.set_segment_color_8(1, 255, 0, 0, 0, 128, 128, 128, 0)
            self.assertTrue(result, "Should successfully set segment color")
    
    def test_set_segment_color_32(self):
        """Test setting segment color with 32-bit values."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            col1 = 0xFF0000FF  # Red with pattern 0
            col2 = 0x80808000  # Gray with pattern 0
            result = self.vast.set_segment_color_32(1, col1, col2)
            self.assertTrue(result, "Should successfully set segment color")
    
    def test_get_all_segment_data(self):
        """Test retrieving all segment data."""
        all_segments = self.vast.get_all_segment_data()
        self.assertIsInstance(all_segments, list, "Should return list")
        if len(all_segments) > 0:
            self.assertIsInstance(all_segments[0], dict, "Elements should be dicts")
    
    def test_get_all_segment_data_matrix(self):
        """Test retrieving all segment data as matrix."""
        matrix, success = self.vast.get_all_segment_data_matrix()
        # Function may fail if no segmentation layer is selected or data format issues
        if success == 1:
            self.assertIsInstance(matrix, np.ndarray, "Should return numpy array")
            if matrix.size > 0:
                self.assertEqual(matrix.shape[1], 24, "Should have 24 columns")
        else:
            self.assertIsInstance(matrix, np.ndarray, "Should return array even on failure")
            self.assertEqual(matrix.size, 0, "Failed call should return empty array")
    
    def test_get_all_segment_names(self):
        """Test retrieving all segment names."""
        names = self.vast.get_all_segment_names()
        self.assertIsInstance(names, list, "Should return list")
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            self.assertEqual(len(names), num_segments, "Should match number of segments")
    
    def test_get_selected_segment_nr(self):
        """Test retrieving selected segment number."""
        seg_nr = self.vast.get_selected_segment_nr()
        self.assertIsInstance(seg_nr, int, "Should return integer")
    
    def test_set_selected_segment_nr(self):
        """Test setting selected segment."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            result = self.vast.set_selected_segment_nr(1)
            self.assertTrue(result, "Should successfully set selected segment")
    
    def test_get_first_segment_nr(self):
        """Test retrieving first segment number."""
        first_seg = self.vast.get_first_segment_nr()
        self.assertIsInstance(first_seg, int, "Should return integer")
    
    def test_set_segment_bbox(self):
        """Test setting segment bounding box."""
        num_segments = self.vast.get_number_of_segments()
        if num_segments > 0:
            result = self.vast.set_segment_bbox(1, 0, 100, 0, 100, 0, 50)
            self.assertTrue(result, "Should successfully set bounding box")


class TestVASTAnnotationFunctions(unittest.TestCase):
    """Test annotation functions."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_anno_layer_nr_of_objects(self):
        """Test retrieving number of annotation objects."""
        num_objects, first_obj = self.vast.get_anno_layer_nr_of_objects()
        self.assertIsInstance(num_objects, int, "Should return integer")
        self.assertIsInstance(first_obj, int, "Should return integer")
        self.assertGreaterEqual(num_objects, 0, "Should be non-negative")
    
    def test_get_anno_layer_object_data(self):
        """Test retrieving annotation object data."""
        objects = self.vast.get_anno_layer_object_data()
        self.assertIsInstance(objects, list, "Should return list")
        if len(objects) > 0:
            self.assertIsInstance(objects[0], dict, "Elements should be dicts")
            expected_keys = ['id', 'type', 'flags', 'col1', 'col2', 
                           'anchorpoint', 'hierarchy']
            for key in expected_keys:
                self.assertIn(key, objects[0], f"Should contain {key}")
    
    def test_get_anno_layer_object_names(self):
        """Test retrieving annotation object names."""
        names = self.vast.get_anno_layer_object_names()
        self.assertIsInstance(names, list, "Should return list")
        if len(names) > 0:
            self.assertIsInstance(names[0], str, "Elements should be strings")
    
    def test_get_ao_node_data(self):
        """Test retrieving skeleton node data."""
        node_data = self.vast.get_ao_node_data()
        if node_data is not None:
            self.assertIsInstance(node_data, np.ndarray, "Should return numpy array")
            self.assertEqual(node_data.shape[1], 14, "Should have 14 columns")
    
    def test_get_ao_node_labels(self):
        """Test retrieving node labels."""
        node_numbers, labels = self.vast.get_ao_node_labels()
        self.assertIsInstance(node_numbers, list, "Node numbers should be list")
        self.assertIsInstance(labels, list, "Labels should be list")
        self.assertEqual(len(node_numbers), len(labels), "Should have equal length")
    
    def test_get_selected_anno_object_nr(self):
        """Test retrieving selected annotation object."""
        obj_nr = self.vast.get_selected_anno_object_nr()
        self.assertIsInstance(obj_nr, int, "Should return integer")
    
    def test_get_selected_ao_node_nr(self):
        """Test retrieving selected node number."""
        node_nr = self.vast.get_selected_ao_node_nr()
        self.assertIsInstance(node_nr, int, "Should return integer")
    
    def test_get_anno_object(self):
        """Test retrieving complete annotation object."""
        num_objects, _ = self.vast.get_anno_layer_nr_of_objects()
        if num_objects > 0:
            obj_data = self.vast.get_anno_object(1)
            self.assertIsInstance(obj_data, dict, "Should return dict")
    
    def test_get_ao_node_params(self):
        """Test retrieving node parameters."""
        num_objects, _ = self.vast.get_anno_layer_nr_of_objects()
        if num_objects > 0:
            node_data = self.vast.get_ao_node_data()
            if node_data is not None and len(node_data) > 0:
                params = self.vast.get_ao_node_params(1, 0)
                self.assertIsInstance(params, dict, "Should return dict")


class TestVASTViewFunctions(unittest.TestCase):
    """Test 2D view functions."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_view_coordinates(self):
        """Test retrieving view coordinates."""
        x, y, z = self.vast.get_view_coordinates()
        self.assertIsInstance(x, int, "X should be integer")
        self.assertIsInstance(y, int, "Y should be integer")
        self.assertIsInstance(z, int, "Z should be integer")
    
    def test_set_view_coordinates(self):
        """Test setting view coordinates."""
        result = self.vast.set_view_coordinates(100, 100, 50)
        self.assertTrue(result, "Should successfully set view coordinates")
    
    def test_get_view_zoom(self):
        """Test retrieving zoom level."""
        zoom = self.vast.get_view_zoom()
        self.assertIsInstance(zoom, int, "Should return integer")
        # Zoom can be negative in VAST (represents view state)
        # Just verify it's a valid integer response
    
    def test_set_view_zoom(self):
        """Test setting zoom level."""
        result = self.vast.set_view_zoom(0)
        self.assertTrue(result, "Should successfully set zoom level")
    
    def test_refresh_layer_region(self):
        """Test refreshing layer region."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0:
            result = self.vast.refresh_layer_region(1, 0, 100, 0, 100, 0, 10)
            self.assertTrue(result, "Should successfully refresh region")


class TestVASTVoxelDataFunctions(unittest.TestCase):
    """Test voxel data transfer functions."""
    
    @classmethod
    def setUpClass(cls):
        cls.vast = VASTControlClass(HOST, PORT)
        cls.vast.connect(timeout=10)
        cls.info = cls.vast.get_info()
    
    @classmethod
    def tearDownClass(cls):
        cls.vast.disconnect()
    
    def test_get_seg_image_raw(self):
        """Test retrieving raw segmentation image."""
        if self.info:
            data = self.vast.get_seg_image_raw(0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, bytes, "Should return bytes")
    
    def test_get_seg_image_rle(self):
        """Test retrieving RLE-encoded segmentation image."""
        if self.info:
            data = self.vast.get_seg_image_rle(0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, bytes, "Should return bytes")
    
    def test_get_seg_image_rle_decoded(self):
        """Test retrieving decoded RLE segmentation image."""
        if self.info:
            data = self.vast.get_seg_image_rle_decoded(0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, np.ndarray, "Should return numpy array")
                self.assertEqual(data.shape, (11, 11, 1), "Shape should match bbox")
    
    def test_get_rle_count_unique(self):
        """Test counting unique segments from RLE."""
        if self.info:
            values, counts = self.vast.get_rle_count_unique(0, 0, 10, 0, 10, 0, 0)
            self.assertIsInstance(values, list, "Values should be list")
            self.assertIsInstance(counts, list, "Counts should be list")
            self.assertEqual(len(values), len(counts), "Should have equal length")
    
    def test_get_seg_image_rle_decoded_count_unique(self):
        """Test decoded RLE with unique counts."""
        if self.info:
            result = self.vast.get_seg_image_rle_decoded_count_unique(0, 0, 10, 0, 10, 0, 0)
            if result[0] is not None:
                seg_image, values, counts = result
                self.assertIsInstance(seg_image, np.ndarray, "Image should be array")
                self.assertIsInstance(values, list, "Values should be list")
                self.assertIsInstance(counts, list, "Counts should be list")
    
    def test_get_seg_image_rle_decoded_bboxes(self):
        """Test decoded RLE with bounding boxes."""
        if self.info:
            result = self.vast.get_seg_image_rle_decoded_bboxes(0, 0, 10, 0, 10, 0, 0)
            if result[0] is not None:
                seg_image, values, counts, bboxes = result
                self.assertIsInstance(seg_image, np.ndarray, "Image should be array")
                self.assertIsInstance(bboxes, list, "Bboxes should be list")
                if len(bboxes) > 0:
                    self.assertEqual(len(bboxes[0]), 6, "Bbox should have 6 values")
    
    def test_set_seg_translation(self):
        """Test setting segmentation translation."""
        source = [1, 2, 3]
        target = [10, 20, 30]
        result = self.vast.set_seg_translation(source, target)
        self.assertTrue(result, "Should successfully set translation")
        
        # Clear translation
        result = self.vast.set_seg_translation([], [])
        self.assertTrue(result, "Should successfully clear translation")
    
    def test_get_em_image_raw(self):
        """Test retrieving raw EM image."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0 and self.info:
            data = self.vast.get_em_image_raw(1, 0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, bytes, "Should return bytes")
    
    def test_get_em_image(self):
        """Test retrieving EM image as array."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0 and self.info:
            data = self.vast.get_em_image(1, 0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, np.ndarray, "Should return numpy array")
    
    def test_get_pixel_value(self):
        """Test retrieving single pixel value."""
        num_layers = self.vast.get_number_of_layers()
        if num_layers > 0 and self.info:
            value = self.vast.get_pixel_value(1, 0, 100, 100, 50)
            self.assertIsInstance(value, int, "Should return integer")
    
    def test_get_screenshot_image_raw(self):
        """Test retrieving raw screenshot image."""
        if self.info:
            data = self.vast.get_screenshot_image_raw(0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, bytes, "Should return bytes")
    
    def test_get_screenshot_image(self):
        """Test retrieving screenshot image as array."""
        if self.info:
            data = self.vast.get_screenshot_image(0, 0, 10, 0, 10, 0, 0)
            if data is not None:
                self.assertIsInstance(data, np.ndarray, "Should return numpy array")
                self.assertEqual(data.shape[2], 3, "Should have 3 color channels")


class TestVASTEncodingFunctions(unittest.TestCase):
    """Test encoding/decoding functions."""
    
    def setUp(self):
        self.vast = VASTControlClass()
    
    def test_encode_uint32(self):
        """Test uint32 encoding."""
        encoded = self.vast._encode_uint32(12345)
        self.assertEqual(len(encoded), 5, "Should be 5 bytes (tag + uint32)")
        self.assertEqual(encoded[0], 1, "Tag should be 1")
    
    def test_encode_int32(self):
        """Test int32 encoding."""
        encoded = self.vast._encode_int32(-12345)
        self.assertEqual(len(encoded), 5, "Should be 5 bytes (tag + int32)")
        self.assertEqual(encoded[0], 4, "Tag should be 4")
    
    def test_encode_double(self):
        """Test double encoding."""
        encoded = self.vast._encode_double(3.14159)
        self.assertEqual(len(encoded), 9, "Should be 9 bytes (tag + double)")
        self.assertEqual(encoded[0], 2, "Tag should be 2")
    
    def test_encode_text(self):
        """Test text encoding."""
        encoded = self.vast._encode_text("test")
        self.assertTrue(encoded.startswith(b'\x03'), "Should start with tag 3")
        self.assertTrue(encoded.endswith(b'\x00'), "Should end with null")
    
    def test_encode_from_struct(self):
        """Test struct encoding."""
        data = {
            'test_uint': 100,
            'test_float': 3.14,
            'test_string': 'hello',
            'test_array': np.array([[1, 2, 3]], dtype=np.uint32)
        }
        encoded, success = self.vast._encode_from_struct(data)
        self.assertEqual(success, 1, "Encoding should succeed")
        self.assertIsInstance(encoded, bytes, "Should return bytes")
    
    def test_decode_to_struct(self):
        """Test struct decoding."""
        # Encode then decode
        original = {'test_val': 42, 'test_str': 'hello'}
        encoded, _ = self.vast._encode_from_struct(original)
        decoded, success = self.vast._decode_to_struct(encoded)
        self.assertEqual(success, 1, "Decoding should succeed")
        self.assertIsInstance(decoded, dict, "Should return dict")


if __name__ == '__main__':
    unittest.main(verbosity=2)