import unittest
import torch
import numpy as np
from F21Stats import F21Stats

class TestF21Stats(unittest.TestCase):
    def setUp(self):
        # Set up some test data
        self.X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        ])
        self.y = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.kernel_sizes = [2, 3]

    def test_calculate_stats_torch_output_shape(self):
        """Test if the output has the correct shape"""
        result = F21Stats.calculate_stats_torch(self.X, self.y, self.kernel_sizes)
        
        # Expected number of features:
        # 2 (total_mean, total_std) + 
        # 4 (mean_skew, std_skew, skew2, min_skew) * len(kernel_sizes)
        expected_features = 2 + 4 * len(self.kernel_sizes)
        
        self.assertEqual(result.shape, (len(self.X), expected_features))

    def test_calculate_stats_torch_basic_stats(self):
        """Test if basic statistics (mean, std) are calculated correctly"""
        result = F21Stats.calculate_stats_torch(self.X, self.y, self.kernel_sizes)
        
        # Test first row's total mean and std
        expected_mean = np.mean(self.X[0])
        expected_std = np.std(self.X[0], ddof=0)
        
        self.assertAlmostEqual(result[0][0], expected_mean, places=5)
        self.assertAlmostEqual(result[0][1], expected_std, places=5)

    def test_calculate_stats_torch_with_empty_input(self):
        """Test handling of empty input"""
        empty_X = np.array([])
        empty_y = np.array([])
        
        with self.assertRaises(Exception):
            F21Stats.calculate_stats_torch(empty_X, empty_y, self.kernel_sizes)

    def test_calculate_stats_torch_with_zero_kernel_size(self):
        """Test handling of invalid kernel size"""
        invalid_kernel_sizes = [0]
        
        with self.assertRaises(Exception):
            F21Stats.calculate_stats_torch(self.X, self.y, invalid_kernel_sizes)

    def test_calculate_stats_torch_values_are_finite(self):
        """Test if all output values are finite (not NaN or inf)"""
        result = F21Stats.calculate_stats_torch(self.X, self.y, self.kernel_sizes)
        self.assertTrue(np.all(np.isfinite(result)))

    """
    def test_calculate_bispectrum_2d(self):
        #Test the bispectrum calculation for 1D data
        data = np.array([1.0, 2.0, 3.0, 4.0])
        expected_shape = (4, 4)  # Since nfft defaults to the length of data

        bispectrum = F21Stats.calculate_bispectrum_2d(data)

        self.assertEqual(bispectrum.shape, expected_shape)

        # Additional checks can be added here based on expected values or properties
        print(bispectrum)
    """
        
    def test_compute_1d_bispectrum_normal(self):
        delta_x = np.random.rand(10)  # Random input
        k1_values = [0.1, 0.2, 0.3]  # Example k1 values
        result = F21Stats.compute_1d_bispectrum(delta_x, k1_values)
        print(f"Printing Bispectrum calculation: Original:\n{delta_x}\n Bispectrum:\n{result}")
        self.assertEqual(result.shape, (len(k1_values),))  # Check output shape

    def test_compute_1d_bispectrum_empty_input(self):
        delta_x = np.array([])  # Empty input
        k1_values = [0.1, 0.2]
        with self.assertRaises(ValueError):  # Expecting an error due to empty input
            F21Stats.compute_1d_bispectrum(delta_x, k1_values)

    def test_compute_1d_bispectrum_single_value(self):
        delta_x = np.array([1.0])  # Single value input
        k1_values = [0.1]
        result = F21Stats.compute_1d_bispectrum(delta_x, k1_values)
        self.assertEqual(result.shape, (1,))  # Check output shape

    def test_compute_1d_bispectrum_invalid_k1(self):
        delta_x = np.random.rand(1024)
        k1_values = [10.0]  # k1 value outside the range
        result = F21Stats.compute_1d_bispectrum(delta_x, k1_values)
        self.assertEqual(result.shape, (1,))  # Check output shape

    def test_compute_1d_bispectrum_negative_k1(self):
        delta_x = np.random.rand(1024)
        k1_values = [-0.1]  # Negative k1 value
        result = F21Stats.compute_1d_bispectrum(delta_x, k1_values)
        self.assertEqual(result.shape, (1,))  # Check output shape

if __name__ == '__main__':
    unittest.main()
