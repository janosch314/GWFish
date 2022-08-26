# Detectors

## Detector classes

The currently supported detector classes are the following:

- `earthDelta` (e.g. Einstein Telescope);
- `earthL` (e.g. LIGO, Virgo, Cosmic Explorer);
- `satellitesolarorbit` (e.g. LISA);
- `lunararray` (e.g. LGWA).

## Detector properties

The properties of each detector are specified in the `GWFish/detectors.yaml` file.
These are strings which will be parsed through an `eval`, so they can include any 
valid `python` expression. The type denoted in parentheses is the one they must be able
to be evaluated into (e.g. `'1e-1*np.pi'` evaluates to a floating point number $\approx 0.314$).

__All detectors__ require:

- __`detector_class`__ (`str`): one of the four aforementioned classes;
- __`fmin`, `fmax`, `df`__ (`float`): minimum and maximum frequency, and frequency spacing 
    (all in Hz): 
    the frequency array is determined by these parameters;
- __`duty_factor`__ (`float` between 0 and 1): the fraction of time the detector is expected to be on for;
- __`plotrange`__ (`tuple` of four `float`s, representing `fmin`, `fmax`, `strain_min`, `strain_max`): 
    x and y limits of a plot of the detector's characteristic noise strain.

__Non-space-based__ `earthDelta`, `earthL` and `lunararray`-type detectors all require:

- __`lat`__ and __`lon`__ (`float`): coordinates of the detector on the surface of the body (Earth/Moon), in radians;
- __`azimuth`__ (`float`): azimuthal angle of the arms, in radians --- for a lunar array, this instead means (???)
- __`psd_data`__ (`str` which is a valid file path): location of a space-separated text file, typically within 
    the folder `GWFish/psd_data/`, containing two columns: frequency (in Hz) and PSD value (in $\text{Hz}^{-1}$).

```{todo}
Unclear meaning of azimuth in the lunar array case --- it likely is the angle
between two of the seismometers, but I'd like to make sure of this.
```

__Earth-bound__ `earthDelta` and `earthL`-type detectors require:

- __`opening_angle`__ (`float`): angle between the detector arms, in radians 
    (should be $\pi/3$ for the triangle, $\pi/2$ for the L);

__Non-Earth-bound__ `lunararray` and `satellitesolarorbit`-type detectors require:

- __`mission_lifetime`__ (`float`): expected mission duration, in seconds.