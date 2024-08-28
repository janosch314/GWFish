import GWFish.modules as gw
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from lal import CreateREAL8TimeSeries, CreateREAL8Vector, DimensionlessUnit

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


def get_fd_signal(parameters, detector_name, waveform_model):
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
            'f_ref': 50.
        }
    waveform_obj = waveform_class(waveform_model, parameters.iloc[0], data_params)
    wave = waveform_obj()
    t_of_f = waveform_obj.t_of_f

    # The waveform is then projected onto the detector taking into account the Earth rotation 
    # by passing at each frequency step the time of the waveform at the detector
    signal = gw.detection.projection(parameters.iloc[0], detector, wave, t_of_f)

    return signal, t_of_f


def get_snr(parameters, network, waveform_model):
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
                'f_ref': 50.
            }
            waveform_obj = waveform_class(waveform_model, parameters.iloc[i], data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f
            signal = gw.detection.projection(parameters.iloc[i], detector, wave, t_of_f)

            snr[detector.name] = np.sqrt(np.sum(gw.detection.SNR(detector, signal)**2))

        snr['network'] = np.sqrt(np.sum([snr[detector.name]**2 for detector in network.detectors]))
        snrs['event_' + str(i)] = snr

    return pd.DataFrame.from_dict(snrs, orient='index')

def make_fft_from_time_series(time_series_input, df, dt, title="Ines_Ludo"):
    '''
    Returns the FFT done through the lal library given a time series. Also returns the frequency array.

    Parameters
    ----------
    time_series_input : array
        Time series data
    df : float
        Frequency step
    dt : float
        Time step
    title : str, optional
        Title of the time series

    Returns
    -------
    tuple
        FFT of the time series and the frequency array
    '''
    dims = len(time_series_input)
    time_series = CreateREAL8Vector(dims)
    time_series.data = time_series_input
    ts = CreateREAL8TimeSeries(title, 1, 0, dt, DimensionlessUnit, dims)
    ts.data = time_series
    fft_dat = gw.fft.fft_lal_timeseries(ts, df).data.data
    freq_range = np.linspace( 0, df * len(fft_dat), len(fft_dat) )
    
    return fft_dat, freq_range

def _fd_phase_correction_and_output_format_from_stain_series(f_, hp, hc, geo_time = 1395964818):
    '''
    Prepares the polarizations for GWFish projection function. Combining 
    the functions "_fd_phase_correction_geocent_time", "_fd_gwfish_output_format" as in LALFD_Waveform class from waveforms.py module.

    Parameters
    ----------
    f_ : array
        Frequency array
    hp : array
        Plus polarization
    hc : array
        Cross polarization
    geo_time : int, optional
        Geocentric time
    
    Returns
    -------
    array
        Polarizations in form (hp, hc)
    '''
    phi_in = np.exp( 1.j * (2 * f_ * np.pi * geo_time) ).T[0]
    fft_dat_plus  = phi_in*np.conjugate( hp )
    fft_dat_cross = phi_in*np.conjugate( hc )

    # GW Fish format for hfp and hfc
    hfp = fft_dat_plus[:, np.newaxis]
    hfc = fft_dat_cross[:, np.newaxis]
    polarizations = np.hstack((hfp, hfc))

    return polarizations

def get_snr(parameters, network, waveform_model = None, series_data = None, long_wavelength_approx = True):
    
    #a routine that only activates if the series_data is provided
    if series_data:
        polarizations, timevector, f_new = series_data
        snrs_series = {}
        for detector in network.detectors:
            detector.frequencyvector = f_new
            args = (parameters, detector, polarizations, timevector)
            signal = gw.detection.projection(*args, long_wavelength_approx = long_wavelength_approx)
            component_SNRs = gw.detection.SNR(detector, signal, frequencyvector=np.squeeze(f_new))
            out_SNR = np.sqrt(np.sum(component_SNRs**2))
            snrs_series[detector.name] = out_SNR

        out_SNR = np.sqrt(np.sum([snrs_series[detector.name]**2 for detector in network.detectors]))
        return out_SNR

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
                'f_ref': 50.
            }
            waveform_obj = waveform_class(waveform_model, parameters.iloc[i], data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f
            signal = gw.detection.projection(parameters.iloc[i], detector, wave, t_of_f)

            snr[detector.name] = np.sqrt(np.sum(gw.detection.SNR(detector, signal)**2))

        snr['network'] = np.sqrt(np.sum([snr[detector.name]**2 for detector in network.detectors]))
        snrs['event_' + str(i)] = snr

    return pd.DataFrame.from_dict(snrs, orient='index')

def get_SNR_from_strains(f_in, hp, hc, network, params, geo_time = 1395964818, long_wavelength_approx = True):

    '''
    Given a set of parameters, polarizations, network, timevector and frequency array, returns the SNR associated to the signal

    Parameters
    ----------
    f_in : array
        Frequency array on which to evaluate the signal
    hp : array
        Plus polarization without geocentric time phase corrections
    hc : array
        Cross polarization without geocentric time phase corrections
    network : gw.detection.DetectorNetwork
        Detector Network object
    params : dict
        Parameters of the event, needs to include ra, dec, psi
    geo_time : int, optional
        Geocentric time
    long_wavelength_approx : bool, optional
        Whether to use the long wavelength approximation or not

    Returns
    -------
    float
        Total signal-to-Noise Ratio 
    '''
        
    polarizations = _fd_phase_correction_and_output_format_from_stain_series(f_in, hp, hc)   
    timevector = np.ones( len(f_in) ) * geo_time

    series_data = (polarizations, timevector, f_in)

    # SNR = get_snr(params, polarizations, detector, timevector, f_in, long_wavelength_approx)
    SNR = get_snr(params, network, series_data = series_data, long_wavelength_approx = long_wavelength_approx)

    return SNR