"""
Functions for LAL FFD of derivatives of time-domain waveforms
"""
import logging

try:
    import lal
except ModuleNotFoundError:
    logging.warning('LAL package is not installed.'+\
                    'Only GWFish waveforms available.')


def fft_lal_timeseries(lal_timeseries, delta_f, f_start=0.):
    """

    f_start: not recommended to change, f_start=0 is in lalsim.SimInspiralFD when calling time-domain waveforms

    Applying a Fourier transform, as in LALSimulation
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L3044
    """

    chirplen = lal_timeseries.data.length
    lal_frequency_series = lal.CreateCOMPLEX16FrequencySeries('FD_H', lal_timeseries.epoch, f_start, delta_f,
                           lal.DimensionlessUnit,int(chirplen / 2 + 1))
    plan = lal.CreateForwardREAL8FFTPlan(chirplen,0)
    lal.REAL8TimeFreqFFT(lal_frequency_series,lal_timeseries,plan)
    return lal_frequency_series


