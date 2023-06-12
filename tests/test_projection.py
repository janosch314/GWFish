import numpy as np
from GWFish.modules.waveforms import TaylorF2
from GWFish.modules.detection import Detector, projection, create_moon_position_interp
from unittest.mock import Mock

BNS_PARAMS = {
    'mass_1': 1.4957673, 
    'mass_2': 1.24276395, 
    'redshift': 0.00980,
    'luminosity_distance': 43.74755446,
    'theta_jn': 2.545065595974997,
    'ra': 3.4461599999999994,
    'dec': -0.4080839999999999,
    'psi': 0.,
    'phase': 0.,
    'geocent_time': 1187008882.4,
    'a_1':0.005136138323169717, 
    'a_2':0.003235146993487445, 
    'lambda_1':368.17802383555687, 
    'lambda_2':586.5487031450857
}

def test_moon_projection():
    # mock = Mock(create_moon_position_interp)
    detector = Detector('LGWA', parameters = [None], fisher_parameters = [None])
    
    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = TaylorF2('TaylorF2', BNS_PARAMS, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f

    signal = projection(
        BNS_PARAMS,
        detector,
        polarizations,
        timevector
    )
    end_time = BNS_PARAMS['geocent_time']
    detector_lifetime = detector.mission_lifetime
    initial_time = end_time - detector_lifetime
    
    assert np.all(timevector < 3786480018)
    assert np.all(signal[timevector > initial_time] != 0)
    assert np.all(signal[timevector < initial_time] == 0j)
    
    signal2 = projection(
        BNS_PARAMS,
        detector,
        polarizations,
        timevector
    )
    
    assert np.all(signal == signal2)

    # mock.assert_called_once()