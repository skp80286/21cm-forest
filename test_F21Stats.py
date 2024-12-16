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

if __name__ == '__main__':
    unittest.main()