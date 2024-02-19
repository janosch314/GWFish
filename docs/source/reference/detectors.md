# Detectors and Networks

(#networks)=
## Networks

```{autodoc2-object} GWFish.modules.detection.Network
render_plugin = "myst"
no_index = true
```

## Detectors

```{autodoc2-object} GWFish.modules.detection.Detector
render_plugin = "myst"
no_index = true
```

### Detector classes

The currently supported detector classes are the following:

- `earthDelta` (e.g. Einstein Telescope);
- `earthL` (e.g. LIGO, Virgo, Cosmic Explorer);
- `satellitesolarorbit` (e.g. LISA);
- `lunararray` (e.g. LGWA).

### Detector properties

The properties of each detector are specified in the `GWFish/detectors.yaml` file.
These are strings which will be parsed through an `eval`, so they can include any 
valid `python` expression. The type denoted in parentheses is the one they must be able
to be evaluated into (e.g. `'1e-1*np.pi'` evaluates to a floating point number $\approx 0.314$).

__All detectors__ require:

- __`detector_class`__ (`str`): one of the four aforementioned classes;
- parameters defining the frequency vector:
    - a __`spacing`__ parameter (`str`), either `geometric` or `linear`, and
    - either __`fmin`, `fmax`, `df`__ (`float`), minimum and maximum frequency, and frequency spacing for the linear spacing option, or __`fmin`, `fmax`__ and __`npoints`__ for the geometric spacing option (all in Hz);
- __`duty_factor`__ (`float` between 0 and 1): the fraction of time the detector is expected to be on for;
- __`plotrange`__ (`tuple` of four `float`s, representing `fmin`, `fmax`, `strain_min`, `strain_max`): 
    x and y limits of a plot of the detector's characteristic noise strain.

__Non-space-based__ `earthDelta`, `earthL` and `lunararray`-type detectors all require:

- __`lat`__ and __`lon`__ (`float`): coordinates of the detector on the surface of the body (Earth/Moon), in radians;
- __`azimuth`__ (`float`): azimuthal angle of the arms, in radians --- for a lunar seismometer array, this instead is the azimuth of the first direction along which the seismometers measure horizontal strain;
- __`psd_data`__ (`str` which is a valid file path): location of a space-separated text file, typically within 
    the folder `GWFish/psd_data/`, containing two columns: frequency (in Hz) and PSD value (in $\text{Hz}^{-1}$).

__Earth-bound__ `earthDelta` and `earthL`-type detectors require:

- __`opening_angle`__ (`float`): angle between the detector arms, in radians 
    (should be $\pi/3$ for the triangle, $\pi/2$ for the L);

__Non-Earth-bound__ `lunararray` and `satellitesolarorbit`-type detectors require:

- __`mission_lifetime`__ (`float`): expected mission duration, in seconds.

(#included-detectors)=
### Included detectors

The following list is automatically generated as a human-readable 
summary of the `GWFish/detectors.yaml` file.

```{include} ../detectors_autogen.inc
```