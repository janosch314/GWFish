import GWFish.modules as gw
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent.parent / 'detectors.yaml'
PSD_PATH = Path(__file__).parent.parent / 'detector_psd'


def get_available_detectors(config=DEFAULT_CONFIG):
    """
    Get the available detectors in the GWFish package, 
    as listed in the .yaml file

    Returns
    -------
    list
        List of available detectors.
    """
    with open(config) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc.keys()

def get_detector_characteristics(detector_name):
    """
    Get the characteristics of a specific detector

    Parameters
    ----------
    detector_name : str
        Name of the detector

    Returns
    -------
    dict
        Dictionary containing the characteristics of the detector
    """
    with open(DEFAULT_CONFIG) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc[detector_name]

def get_detector_psd(detector_name):
    """
    Get the Power Spectral Density of a detector

    Parameters
    ----------
    detector_name : str
        Name of the detector

    Returns
    -------
    numpy.ndarray
        Power Spectral Density of the detector
    """
    with open(PSD_PATH / f'{detector_name}_psd.txt', 'rb') as f:
        return np.loadtxt(f, usecols=[0, 1])
    
def add_new_detector(detector_name, dictionary, config=DEFAULT_CONFIG):
    """
    Create a .yaml file from a dictionary

    Parameters
    ----------
    detector_name : str
        Name of the detector
    dictionary : dict
        Dictionary to be saved to the .yaml file
    config : str, optional
    """
    # check all the necessary keys are present
    keys = ['lat', 'lon', 'opening_angle', 'azimuth', 'psd_data', 'duty_factor', 'detector_class',
            'fmin', 'fmax', 'spacing', 'df', 'npoints']
    for key in keys:
        if key not in dictionary.keys():
            raise KeyError(f"Key {key} must be specified in dictionary")
    if 'plot_range' not in list(dictionary.values()):
        dictionary['plot_range'] = '3, 1000, 1e-25, 1e-20'

    new_detector = {detector_name: dictionary}
    with open(config, 'a') as f:
        yaml.dump(new_detector, f, indent=8)


def get_fd_signal(parameters, detector_name, waveform_model, f_ref=gw.waveforms.DEFAULT_F_REF):
    """
    Get the frequency domain signal projected onto the detector from a waveform model 
    and a set of parameters

    Parameters
    ----------
    parameters : pandas.DataFrame
        DataFrame containing the parameters of the event
    network : gw.DetectorNetwork

    waveform_model : str

    Returns
    -------
    numpy.ndarray
        Signal projected onto the detector
    """

    # The waveform model can be accessed through the waveform_class attribute,
    # which requires the waveform_model and the data_params and the parameters of the event
    detector = gw.detection.Detector(detector_name)
    waveform_class = gw.waveforms.LALFD_Waveform
    data_params = {
            'frequencyvector': detector.frequencyvector,
            'f_ref': f_ref
        }
    waveform_obj = waveform_class(waveform_model, parameters.iloc[0], data_params)
    wave = waveform_obj()
    t_of_f = waveform_obj.t_of_f

    # The waveform is then projected onto the detector taking into account the Earth rotation 
    # by passing at each frequency step the time of the waveform at the detector
    signal = gw.detection.projection(parameters.iloc[0], detector, wave, t_of_f)

    return signal, t_of_f


def get_snr(parameters, network, waveform_model, f_ref=gw.waveforms.DEFAULT_F_REF):
    """
    Get the Signal-to-Noise Ratio of single detectors and combined in a network

    Parameters
    ----------
    parameters : pandas.DataFrame
        DataFrame containing the parameters of the event
    network : gw.DetectorNetwork
        Detector network
    waveform_model : str
        Waveform model

    Returns
    -------
    pandas.DataFrame
        Signal-to-Noise Ratio in individual detectors and in the network
    """
    waveform_class = gw.waveforms.LALFD_Waveform

    nsignals = len(parameters)
    
    # The SNR is then computed by taking the norm of the signal projected onto the detector
    # and dividing by the noise of the detector
    snrs = {}
    for i in range(nsignals):
        snr = {}
        for detector in network.detectors:
            data_params = {
                'frequencyvector': detector.frequencyvector,
                'f_ref': f_ref
            }
            waveform_obj = waveform_class(waveform_model, parameters.iloc[i], data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f
            signal = gw.detection.projection(parameters.iloc[i], detector, wave, t_of_f)

            snr[detector.name] = np.sqrt(np.sum(gw.detection.SNR(detector, signal)**2))

        snr['network'] = np.sqrt(np.sum([snr[detector.name]**2 for detector in network.detectors]))
        snrs['event_' + str(i)] = snr

    return pd.DataFrame.from_dict(snrs, orient='index')

 