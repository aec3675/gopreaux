import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch


class TestGP3D:
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_gp3d):
        self.phasemin = -20
        self.phasemax = 50
        self.gp = mock_gp3d
        self.phase_grid = np.arange(0.5, 5.0, 0.1)
        self.wl_grid = np.log10(np.arange(2000.0, 8000.0, 100.0))
    
    def test_build_samples(self):
        """Test build_samples method"""
        ### Test all boolean combinations of log_transform and use_flux
        with patch("caat.GP.GP._process_dataset", Mock(
            return_value=(
                np.asarray([0.5, 3.5]), 
                np.asarray([-1.0, 3.0]),
                np.asarray([0.1, 0.1]), 
                np.asarray([3.5, 3.5])
            )
        )):

            phases, wls, mags, err_grid = (
                self.gp._build_samples('B')
            )

            assert all([isinstance(var, np.ndarray) for var in [phases, mags, wls, err_grid]])
            assert len(phases) == len(mags) == len(wls) == len(err_grid)

    def test_process_dataset(self):
        """Test process_dataset method"""
        template_df = self.gp._process_dataset()
        assert isinstance(template_df, pd.DataFrame)

    def test_median_grid(self, mock_datacube):
        """Test construct_median_grid method"""
        phase_grid, wl_grid, mag_grid, err_grid = (
            self.gp._construct_median_grid(
                self.phasemin,
                self.phasemax,
                ['B'],
                mock_datacube,
                log_transform=22,
                plot=False,
            )
        )

        assert mag_grid.shape == (len(phase_grid), len(wl_grid))
        assert err_grid.shape == (len(phase_grid), len(wl_grid))

    def test_polynomial_grid(self, mock_datacube):
        """Test construct_polynomial_grid method"""
        phase_grid, wl_grid, mag_grid, err_grid = (
            self.gp._construct_polynomial_grid(
                self.phasemin,
                self.phasemax,
                ['B'],
                mock_datacube,
                log_transform=22,
                plot=False,
            )
        )

        assert mag_grid.shape == (len(phase_grid), len(wl_grid))
        assert err_grid.shape == (len(phase_grid), len(wl_grid))

    def test_subtract_from_grid(self, mock_sn):
        """Test subtract_from_grid method"""
        residuals = self.gp._subtract_data_from_grid(
            mock_sn,
            ['B'],
            self.phase_grid,
            self.wl_grid,
            np.random.random((len(self.phase_grid), len(self.wl_grid))),
            np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
        )

        assert isinstance(residuals, pd.DataFrame)

    def test_run_gp_full_sample_without_specifying_subtract(self):
        """Should raise an Exception when running GP without specifying a subtract method"""
        with pytest.raises(Exception, match=r'Must toggle either .*'):
            self.gp.run_gp_on_full_sample(
                plot=False,
            )

    def test_run_gp_individually_without_specifying_subtract(self):
        """Should raise an Exception when running GP without specifying a subtract method"""
        with pytest.raises(Exception, match=r'Must toggle either .*'):
            self.gp.run_gp_individually(
                plot=False,
            )

    def test_build_test_wavelength_phase_grid_from_photometry(self, mock_datacube):
        """Test build_test_wavelength_phase_grid_from_photometry method"""
        (
            x, 
            y, 
            wl_inds_fitted, 
            phase_inds_fitted, 
            min_phase
        ) = self.gp._build_test_wavelength_phase_grid_from_photometry(
            mock_datacube["LogShiftedWavelength"].values,
            mock_datacube["LogPhase"].values,
            self.wl_grid,
            self.phase_grid,
        )

        assert len(x) == len(y)
        assert isinstance(wl_inds_fitted, list | np.ndarray)
        assert isinstance(phase_inds_fitted, list | np.ndarray)
        assert isinstance(min_phase, float | None)

    def test_optimize_hyperparameters(self):
        """Test optimize_hyperparameters method"""
        class MockGaussianProcessRegressor:
            class kernel_:
                theta = [1.0, 1.0]
            
            def fit(*args, **kwargs):
                pass

        with patch(
            "caat.GP3D.GP3D._construct_polynomial_grid", Mock(
                return_value=(
                    self.phase_grid, 
                    self.wl_grid,
                    np.random.random((len(self.phase_grid), len(self.wl_grid))),
                    np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
                )
            )
        ), patch(
            "caat.GP3D.GP3D._subtract_data_from_grid", Mock(
                return_value=pd.DataFrame(
                    [
                        {
                            "Filter": "B",
                            "Phase": -15.0,
                            "Wavelength": 5000.0,
                            "MagResidual": 0.1,
                            "MagErr": 0.1,
                            "Mag": -0.5,
                            "Nondetection": False,
                        },
                        {
                            "Filter": "B",
                            "Phase": -15.0,
                            "Wavelength": 5000.0,
                            "MagResidual": 0.1,
                            "MagErr": 0.1,
                            "Mag": -0.5,
                            "Nondetection": False,
                        }
                    ]
                )
            )
        ), patch(
            "sklearn.gaussian_process.GaussianProcessRegressor", MockGaussianProcessRegressor
        ):
            self.gp.kernel.set_params = Mock()
            kernel_params = self.gp.optimize_hyperparams(subtract_polynomial=True)
            assert isinstance(kernel_params, list)
            self.gp.kernel.set_params.assert_called_once()

    # def test_run_gp3d(self):

    #     gp = GP3D(sncollection, kernel)
    #     gaussian_processes, phase_grid, kernel_params, wl_grid = (
    #         gp.run_gp(
    #             ['B', 'g', 'V'],
    #             -20,
    #             50,
    #             log_transform=30,
    #             fit_residuals=True,
    #             set_to_normalize=sncollection.sne,
    #             subtract_polynomial=True,
    #             use_fluxes=use_flux
    #         )
    #     )

    #     assert len(gaussian_processes) > 0 and len(phase_grid) > 0 and len(kernel_params) > 0 and len(wl_grid) > 0
    #     assert gaussian_processes[0].shape <= (len(phase_grid), len(wl_grid))

    def test_interpolate_grid_fills_nans(self):
        """interpolate_grid should replace NaNs between valid values (requires >4 valid points)"""
        # filter_window must be > polyorder (3), so use 5; need >4 non-NaN values per row
        row = np.array([1.0, 1.5, 2.0, np.nan, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=float)
        grid = np.vstack([row, row + 1.0])
        interp_array = np.linspace(0, 1, grid.shape[1])
        result = self.gp.interpolate_grid(grid.copy(), interp_array, filter_window=5)
        # The NaN at index 3 is between valid values and should be filled
        assert not np.isnan(result[0, 3])
        assert not np.isnan(result[1, 3])

    def test_interpolate_grid_preserves_endpoints(self):
        """interpolate_grid should leave NaNs outside the range of valid values"""
        grid = np.array([[np.nan, 1.0, 2.0, np.nan]], dtype=float)
        interp_array = np.linspace(0, 1, 4)
        result = self.gp.interpolate_grid(grid.copy(), interp_array, filter_window=3)
        # Endpoints outside valid data range stay NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 3])

    def test_interpolate_grid_all_nan_row_unchanged(self):
        """Rows with all NaNs should remain all NaN"""
        grid = np.array([[np.nan, np.nan, np.nan]], dtype=float)
        interp_array = np.linspace(0, 1, 3)
        result = self.gp.interpolate_grid(grid.copy(), interp_array, filter_window=3)
        assert np.all(np.isnan(result))

    def test_process_dataset_with_set_to_normalize(self):
        """_process_dataset with set_to_normalize returns a DataFrame"""
        with patch("caat.GP.GP._process_dataset", Mock(
            return_value=(
                np.asarray([0.5, 3.5]),
                np.asarray([-1.0, 3.0]),
                np.asarray([0.1, 0.1]),
                np.asarray([3.5, 3.5])
            )
        )):
            template_df = self.gp._process_dataset(
                set_to_normalize=self.gp.set_to_normalize
            )
        assert isinstance(template_df, pd.DataFrame)
        assert set(template_df.columns) >= {"Phase", "Wavelength", "Mag", "MagErr"}

    def test_build_samples_returns_empty_arrays_when_no_data(self):
        """_build_samples should return empty arrays when _process_dataset returns nothing"""
        with patch("caat.GP.GP._process_dataset", Mock(
            return_value=(
                np.asarray([]),
                np.asarray([]),
                np.asarray([]),
                np.asarray([])
            )
        )):
            phases, wls, mags, err_grid = self.gp._build_samples('B')

        assert len(phases) == 0
        assert len(wls) == 0
        assert len(mags) == 0
        assert len(err_grid) == 0

    def test_build_test_wavelength_phase_grid_returns_empty_when_no_phases_in_range(
        self, mock_datacube
    ):
        """Returns empty lists when no phases in the grid match the measured phases"""
        # Put measured phases far outside the wl_grid range
        measured_phases = np.asarray([100.0, 101.0])  # >> phase_grid max
        result = self.gp._build_test_wavelength_phase_grid_from_photometry(
            mock_datacube["LogShiftedWavelength"].values,
            measured_phases,
            self.wl_grid,
            self.phase_grid,
        )
        x, y, wl_inds_fitted, phase_inds_fitted, min_phase = result
        assert len(x) == 0
        assert len(y) == 0
        assert min_phase is None

    def test_sample_predicted_sed_shape(self):
        """_sample_predicted_sed returns array with same shape as input"""
        mean = np.random.random((10, 5))
        std = np.random.random((10, 5)) * 0.1
        result = self.gp._sample_predicted_sed(mean, std)
        assert result.shape == mean.shape

    def test_sample_predicted_sed_within_bounds(self):
        """Sampled SED should be within ±1 sigma of the mean"""
        mean = np.zeros((20, 20))
        std = np.ones((20, 20))
        result = self.gp._sample_predicted_sed(mean, std)
        assert np.all(result >= mean - std)
        assert np.all(result <= mean + std)

    def test_smooth_predicted_model_shape_preserved(self):
        """_smooth_predicted_model should preserve the shape of the input array"""
        model = np.random.random((5, 10))
        result = self.gp._smooth_predicted_model(model, window_size=3)
        assert result.shape == model.shape

    def test_smooth_predicted_model_transpose(self):
        """_smooth_predicted_model with transpose=True should still return original shape"""
        model = np.random.random((5, 10))
        result = self.gp._smooth_predicted_model(model, window_size=3, transpose=True)
        assert result.shape == model.shape

    def test_smooth_predicted_model_even_window_becomes_odd(self):
        """Even window_size should be incremented to odd without raising an error"""
        model = np.random.random((4, 8))
        result = self.gp._smooth_predicted_model(model, window_size=4)
        assert result.shape == model.shape

    def test_optimize_hyperparams_raises_without_subtract_flag(self):
        """optimize_hyperparams should raise when neither subtract flag is set"""
        with patch("caat.GP3D.GP3D._process_dataset", Mock(return_value=pd.DataFrame(
            {"Phase": [], "Wavelength": [], "Mag": [], "MagErr": []}
        ))):
            with pytest.raises(Exception, match=r'Must toggle either .*'):
                self.gp.optimize_hyperparams()

    def test_subtract_data_from_grid_skips_nan_grid_values(self, mock_sn):
        """Residuals for data points where mag_grid is NaN should be skipped"""
        nan_mag_grid = np.full(
            (len(self.phase_grid), len(self.wl_grid)), np.nan
        )
        residuals = self.gp._subtract_data_from_grid(
            mock_sn,
            ['B'],
            self.phase_grid,
            self.wl_grid,
            nan_mag_grid,
            np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
        )
        assert isinstance(residuals, pd.DataFrame)
        assert len(residuals) == 0

    def test_subtract_data_from_grid_columns(self, mock_sn):
        """Residuals DataFrame should contain the expected columns"""
        mag_grid = np.random.random((len(self.phase_grid), len(self.wl_grid)))
        residuals = self.gp._subtract_data_from_grid(
            mock_sn,
            ['B'],
            self.phase_grid,
            self.wl_grid,
            mag_grid,
            np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
        )
        assert isinstance(residuals, pd.DataFrame)
        if len(residuals) > 0:
            expected_cols = {"Filter", "Phase", "Wavelength", "MagResidual", "MagErr", "Mag", "Nondetection"}
            assert expected_cols <= set(residuals.columns)