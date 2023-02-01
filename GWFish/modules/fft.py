"""
Functions for LAL FFD of derivatives of time-domain waveforms
"""
import lal

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

# The functions below are not used in practice, for now

def fft(hh, dt, t_start, t_end, roll_off = 0.2):
    """
    Perform FFT to convert the data from time domain to frequency domain.
    Roll-off is specified for the Tukey window in [s].
    """
    alpha = 2 * roll_off / (t_end - t_start)
    window = tukey(len(hh), alpha=alpha)
    hh_tilde = np.fft.rfft(hh * window)
    hh_tilde /= 1/dt
    ff = np.linspace(0, (1/dt) / 2, len(hh_tilde))
    # Future: here, one can check if frequency resolution and minimum frequency requested are
    # lower than waveform time span. Resolution freq: warning. Minimum freq: ValueError.
    return hh_tilde, ff

def ifft(hh_tilde, df):
    """
    Perform inverse FFT to convert the data from frequency domain to time domain.
    """
    return np.fft.ifft(hh_tilde) * df
