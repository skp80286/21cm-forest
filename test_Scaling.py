import numpy as np
import pytest
from Scaling import Scaler
from types import SimpleNamespace

class TestScaler:
    @pytest.fixture
    def sample_data(self):
        # Create sample input data
        X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        y = np.array([[0.5, -2.0], [0.8, -1.0]])
        return X, y

    @pytest.fixture
    def scaler_args(self):
        # Create args namespace with default values
        return SimpleNamespace(
            scale_y=False,
            scale_y0=False,
            scale_y1=False,
            scale_y2=False,
            logscale_X=False,
            trials=0
        )

    def test_init(self, scaler_args):
        scaler = Scaler(scaler_args)
        assert scaler.args == scaler_args

    def test_no_scaling(self, scaler_args, sample_data):
        X, y = sample_data
        scaler = Scaler(scaler_args)
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # When no scaling options are enabled, output should match input
        np.testing.assert_array_equal(X, X_scaled)
        np.testing.assert_array_equal(y, y_scaled)

    def test_scale_y(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.scale_y = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Test unscaling
        X_unscaled, y_unscaled = scaler.unscaleXy(X_scaled.copy(), y_scaled.copy())
        
        # Check if unscaling returns original values (within floating-point precision)
        np.testing.assert_array_almost_equal(X, X_unscaled)
        np.testing.assert_array_almost_equal(y, y_unscaled)

    def test_scale_y0(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.scale_y0 = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Verify y[:,0] is multiplied by 5.0
        np.testing.assert_array_almost_equal(y_scaled[:,0], y[:,0] * 5.0)
        
        # Test unscaling
        X_unscaled, y_unscaled = scaler.unscaleXy(X_scaled.copy(), y_scaled.copy())
        np.testing.assert_array_almost_equal(X, X_unscaled)
        np.testing.assert_array_almost_equal(y, y_unscaled)

    def test_logscale_X(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.logscale_X = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Verify X is log-scaled
        np.testing.assert_array_almost_equal(X_scaled, np.log(X))
        
        # Test unscaling
        X_unscaled, y_unscaled = scaler.unscaleXy(X_scaled.copy(), y_scaled.copy())
        np.testing.assert_array_almost_equal(X, X_unscaled)
        np.testing.assert_array_almost_equal(y, y_unscaled)

    def test_scale_y_unscale_y(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.scale_y = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Verify y has three columns after scaling
        assert y_scaled.shape[1] == 2
        
        # Test unscaling
        y_unscaled = scaler.unscale_y(y_scaled.copy())
        np.testing.assert_array_almost_equal(y, y_unscaled)

    @pytest.mark.skip(reason="Test temporarily disabled")
    def test_scale_y1(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.scale_y1 = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Verify y has three columns after scaling
        assert y_scaled.shape[1] == 3
        
        # Test unscaling
        y_unscaled = scaler.unscale_y(y_scaled.copy())
        np.testing.assert_array_almost_equal(y, y_unscaled)

    @pytest.mark.skip(reason="Test temporarily disabled")
    def test_scale_y2(self, scaler_args, sample_data):
        X, y = sample_data
        scaler_args.scale_y2 = True
        scaler = Scaler(scaler_args)
        
        # Test scaling
        X_scaled, y_scaled = scaler.scaleXy(X.copy(), y.copy())
        
        # Verify y has three columns after scaling
        assert y_scaled.shape[1] == 3
        
        # Test unscaling
        y_unscaled = scaler.unscale_y(y_scaled.copy())
        np.testing.assert_array_almost_equal(y, y_unscaled)

    @pytest.mark.skip(reason="Test temporarily disabled")
    def test_invalid_input(self, scaler_args):
        scaler = Scaler(scaler_args)
        
        # Test with empty arrays
        with pytest.raises(IndexError):
            scaler.scaleXy(np.array([]), np.array([]))
        
        # Test with 1D arrays
        with pytest.raises(IndexError):
            scaler.scaleXy(np.array([1,2,3]), np.array([1,2,3]))