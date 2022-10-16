import numpy as np
from GWFish.modules.waveforms import hphc_amplitudes
from GWFish.modules.detection import Detector, projection

def test_max_f_cutoff_170817():
    
    params = {
        'mass_1': 1.4, 
        'mass_2': 1.4, 
        'redshift': 0.01,
        'luminosity_distance': 40,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
        'max_frequency': 400,
    }
    
    detector = Detector('ET', parameters = [None], fisher_parameters = [None])
    
    hphc, t_of_f = hphc_amplitudes('gwfish_TaylorF2', params, detector.frequencyvector)
    
    assert hphc[-1, 0] == 0j
    assert hphc[-1, 1] == 0j
    assert hphc[0, 0] != 0j
    assert hphc[0, 1] != 0j
    
    params.pop('max_frequency')
    
    hphc, t_of_f = hphc_amplitudes('gwfish_TaylorF2', params, detector.frequencyvector)
    
    assert hphc[-1, 0] != 0j
    assert hphc[-1, 1] != 0j
    
def test_max_f_cutoff_signal_duration():

    # 170817-like parameters, with masses switched to BWD-like ones.
    params = {
        'mass_1': .6, 
        'mass_2': .6, 
        'redshift': 0.01,
        'luminosity_distance': 40,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
        'max_frequency': .2,
    }

    # LGWA mission duration is 10 years
    detector = Detector('LGWA', parameters = [None], fisher_parameters = [None])
    
    # if the BWD merges at 0.2 Hz (say), then the relevant part of the waveform is 
    # at best the one from 0.13Hz to 0.2Hz, since the BWD will take ~10yr to get
    # through that interval.
    
    polarizations, timevector = hphc_amplitudes('gwfish_TaylorF2', params, detector.frequencyvector)

    signal = projection(
        params,
        detector,
        polarizations,
        timevector
    )
    
    # if the signal is all zero that's a problem
    assert not np.all(signal == 0.j)

    signal_nonzero_indices, = np.where(signal[:, 0])
    
    # print(signal)
    # print(nonzero_times)

    # the indices at which the signal is nonzero should be "in the middle",
    # not starting nor ending at the edge frequencies
    assert signal_nonzero_indices[0] > 0
    assert signal_nonzero_indices[-1] < timevector.shape[0]
    
    # print(len(signal_nonzero_indices))
    # print(timevector.shape)
    
    # hardest check: the region of the signal for which the 
    nonzero_times = timevector[signal_nonzero_indices, 0]
    delta_t = nonzero_times[1] - nonzero_times[0]
    
    # time delta corresponding to the lowest frequency spacing 
    # in which the signal is considered to be nonzero
    # it should be smaller than one year
    assert delta_t < 3e7

    assert np.isclose(
        detector.mission_lifetime, 
        nonzero_times[-1] - nonzero_times[0],
        atol = delta_t,
        rtol = 0
    )
