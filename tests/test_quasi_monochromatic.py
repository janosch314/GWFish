from datetime import timedelta
from hypothesis import given, settings
from hypothesis import HealthCheck, example, given, settings


from GWFish.modules.horizon import MIN_REDSHIFT, compute_SNR, horizon
from GWFish.modules.detection import Detector, Network, projection, in_band_window
from GWFish.modules.waveforms import LALFD_Waveform

import numpy as np
import pytest

from .test_horizon import extrinsic

DETECTORS = ['LGWA', 'LISA']
@pytest.fixture(
    scope='module',    
    params=DETECTORS,
    ids=DETECTORS,
)
def low_frequency_detector(request):
    # defined here so that the ephemeris are not recomputed every iteration
    return Detector(request.param)

@given(extrinsic())
@settings(max_examples=100, deadline=None)
def test_quasi_monochromatic_horizon(low_frequency_detector, extrinsic):
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic

    # very light WD binary, will give errors unless it's properly treated
    # since it's slowly evolving 
    params = {
        'mass_1': 0.2,
        'mass_2': 0.2,
        'theta_jn': theta_jn, 
        'dec': declination, 
        'ra': right_ascension, 
        'psi': polarization, 
        'phase': phase, 
        'geocent_time': gps_time,
        'max_frequency_cutoff': 0.1,
        'redshift': 0.,
        'luminosity_distance': 1.,
    }
    
    
    data_params = {
        'frequencyvector': low_frequency_detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = LALFD_Waveform('TaylorF2', params, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f
    
    signal, new_t, new_f = projection(
        params,
        low_frequency_detector,
        polarizations,
        timevector,
        redefine_tf_vectors=True
    )
    
    # the minimum gps time is 1992, so with 10-year missions at most we should 
    # not be able to go below 1980
    assert np.all(new_t > 0)
    
    assert len(np.nonzero(signal[:, 0])[0]) > 50


def test_in_band_window_no_constraints():
    initial_timevector = np.linspace(0, 10, 10)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 10), 
        detector_lifetime=None,
        max_frequency_cutoff=None
    )
    
    assert np.allclose(t, initial_timevector)
    assert s.start == 0
    assert s.stop == 10

def test_in_band_window_with_lifetime():
    initial_timevector = np.linspace(0, 10, 10)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 10), 
        detector_lifetime=2.9, # if this was 3, we'd get an extra point
        max_frequency_cutoff=None
    )
    
    assert np.allclose(t, initial_timevector)
    assert s.start == 7
    assert s.stop == 10

def test_in_band_window_with_max_frequency():
    initial_timevector = np.linspace(0, 10, 11)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 11), 
        detector_lifetime=None,
        max_frequency_cutoff=0.5,
        redefine_timevector=True,
    )
    
    assert np.allclose(t, initial_timevector + 5.)
    assert s.start == 0
    assert s.stop == 5

def test_in_band_window_with_max_frequency_and_lifetime():
    initial_timevector = np.linspace(0, 10, 11)
    # in this context, the max_frequency_cutoff is 0.5, meaning that the upper 
    # end of the in-band window should be at "time 10", which is the set "GPS time"
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 11), 
        detector_lifetime=3., 
        max_frequency_cutoff=0.5,
        redefine_timevector=True,
    )
    
    assert np.allclose(t, initial_timevector+5.)
    assert s.start == 2
    assert s.stop == 5

@pytest.mark.parametrize('max_frequency_cutoff', [0.5, None])
def test_in_band_window_lifetime_under_min_frequency(max_frequency_cutoff):
    initial_timevector = np.linspace(0, 10, 10)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 10), 
        detector_lifetime=20, 
        max_frequency_cutoff=max_frequency_cutoff,
    )
    
    assert s.start == 0
    if max_frequency_cutoff is None:
        assert s.stop == 10
    else:
        assert s.stop == int(10*max_frequency_cutoff)

@pytest.mark.parametrize('max_frequency_cutoff', [0.5, None])
def test_in_band_window_does_not_modify_timevector(max_frequency_cutoff):
    initial_timevector = np.linspace(0, 10, 10)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 10), 
        detector_lifetime=20, 
        max_frequency_cutoff=max_frequency_cutoff,
    )

    assert np.allclose(initial_timevector, np.linspace(0, 10, 10)) # should not have been modified

@pytest.mark.parametrize('max_frequency_cutoff', [0.5, 0.45])
def test_in_band_window_very_short_signal(max_frequency_cutoff):
    initial_timevector = np.linspace(0, 10, 10)
    s, t = in_band_window(
        timevector=initial_timevector, 
        frequencyvector=np.linspace(0, 1, 10), 
        detector_lifetime=0.001, 
        max_frequency_cutoff=max_frequency_cutoff,
    )

    assert s.start == s.stop