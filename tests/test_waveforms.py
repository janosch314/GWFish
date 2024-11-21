import numpy as np

from GWFish.modules.detection import Detector, projection
from GWFish.modules.waveforms import TaylorF2

import pytest

ATOL = 1e-30

# @pytest.mark.xfail
def test_max_f_cutoff_170817():
    
    params = {
        'mass_1_source': 1.4, 
        'mass_2_source': 1.4, 
        'redshift': 0.01,
        'luminosity_distance': 40,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
        'max_frequency_cutoff': 400,
    }
    
    detector = Detector('ET')

    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = TaylorF2('TaylorF2', params, data_params)
    hphc = waveform_obj()
    t_of_f = waveform_obj.t_of_f
    
    proj_with_cutoff = projection(params, detector, hphc, t_of_f, (0, 0, 0))
    
    # the signal should be cut off at high frequency, therefore 
    # the last element should be zero, while at low frequency it should
    # be nonzero.
    assert np.allclose(proj_with_cutoff[-1, :], 0j, atol=ATOL)
    assert not np.allclose(proj_with_cutoff[0, :], 0j, atol=ATOL)
    
    params.pop('max_frequency_cutoff')
    
    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = TaylorF2('TaylorF2', params, data_params)
    hphc = waveform_obj()
    t_of_f = waveform_obj.t_of_f
    
    proj_no_cutoff = projection(params, detector, hphc, t_of_f)
    
    assert not np.allclose(proj_no_cutoff[-1, :], 0j, atol=ATOL)
    
@pytest.mark.parametrize('redefine_tf_vectors', [True, False])
def test_max_f_cutoff_signal_duration(redefine_tf_vectors):

    # 170817-like parameters, with masses switched to BWD-like ones.
    params = {
        'mass_1_source': .6, 
        'mass_2_source': .6, 
        'redshift': 0.01,
        'luminosity_distance': 40,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
        'max_frequency_cutoff': .2,
    }

    # LGWA mission duration is 10 years
    detector = Detector('LGWA')
    
    # if the BWD merges at 0.2 Hz (say), then the relevant part of the waveform is 
    # at best the one from 0.13Hz to 0.2Hz, since the BWD will take ~10yr to get
    # through that interval.
    
    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = TaylorF2('TaylorF2', params, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f

    if redefine_tf_vectors:
        signal, timevector, frequencyvector = projection(
            params,
            detector,
            polarizations,
            timevector,
            center=(0, 0, 0),
            redefine_tf_vectors=True
        )
    
    else:
        signal = projection(
            params,
            detector,
            polarizations,
            timevector,
            center=(0, 0, 0),
        )
    
    # if the signal is all zero that's a problem
    assert not np.all(signal == 0.j)

    signal_nonzero_indices, = np.where(signal[:, 0])

    # the indices at which the signal is nonzero should be "in the middle",
    # not starting nor ending at the edge frequencies
    assert signal_nonzero_indices[0] > 0
    assert signal_nonzero_indices[-1] < timevector.shape[0]
    
    # hardest check: the region of the signal for which the
    # waveform is nonzero should last the correct amount of time
    nonzero_times = timevector[signal_nonzero_indices]
    
    # we define a margin of error based on the discretization of the time vector
    # in the region where the signal is nonzero
    
    delta_t = (
        nonzero_times[1] - nonzero_times[0]
    ) + (
        nonzero_times[-1] - nonzero_times[-2]
    )
    
    # time delta corresponding to the lowest frequency spacing 
    # in which the signal is considered to be nonzero
    # it should be smaller than one year
    assert delta_t < 3e7
    
    if redefine_tf_vectors:
        # expect this to be close to 1000
        assert len(nonzero_times) > 900

    assert np.isclose(
        detector.mission_lifetime, 
        nonzero_times[-1] - nonzero_times[0],
        atol = delta_t,
        rtol = 0
    )
    
def compute_match(h1, h2, f, psd):
    
    return (
        np.trapz(h1* np.conj(h2) / psd, x=f) 
        / np.sqrt(
            np.trapz(abs(h1)**2 / psd, x=f) *
            np.trapz(abs(h2)**2 / psd, x=f)
        )
    )

def test_mismatch_with_param_variation():
    
    params = {
        'mass_ratio': 0.99, 
        'luminosity_distance': 40,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
        'chirp_mass_detector': 2.8/2**(6/5), 
    }
    
    detector = Detector('ET')

    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = TaylorF2('TaylorF2', params, data_params)
    hphc = waveform_obj()
    t_of_f = waveform_obj.t_of_f
    proj = projection(params, detector, hphc, t_of_f, (0, 0, 0))

    for name in params.keys():
        new_params = params.copy()
        if name == 'chirp_mass_detector':
            new_params[name] = params[name] + 1e-8
        else:
            new_params[name] = params[name] + 1e-5
        waveform_obj_2 = TaylorF2('TaylorF2', new_params, data_params)
        hphc_2 = waveform_obj_2()
        t_of_f_2 = waveform_obj_2.t_of_f
        proj_2 = projection(new_params, detector, hphc_2, t_of_f_2, (0, 0, 0))

        f = np.squeeze(detector.frequencyvector)
        assert compute_match(
            proj[:, 0], 
            proj_2[:, 0], 
            f, 
            detector.components[0].Sn(f)
        ).real > 0.999