import numpy as np
from GWFish.modules.waveforms import hphc_amplitudes
from GWFish.modules.detection import Detector

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
    
