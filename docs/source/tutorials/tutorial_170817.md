# Estimating {term}`SNR` and errors for a single signal

This tutorial will show how to compute the {term}`SNR` and Fisher matrix errors for a 
compact object signal with known parameters.
For more details on the mathematics of how the errors are obtained, see the 
[Fisher matrix reference](../explanation/fisher_matrix.md).

## What we want to do

We will study a GW170817-like signal, as would be seen by a network of Einstein Telescope and two Cosmic Explorers.
GWFish can give us estimates of the overall {term}`SNR` it would have, 
as well as the errors in the estimates of the signal parameters.

## Population file

The outcome of this section is a [Pandas `DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing several columns, each corresponding
to one of the parameters characterizing a {term}`CBC` signal.
The units and conventions for these parameters are outlined in the [reference](../reference/parameters_units.md).

Assuming we already know what the values of the parameters should be, we can generate this dataframe as follows:

```python
>>> import pandas as pd
>>> import numpy as np

>>> param_dict = {
...     'mass_1': 1.4 * (1 + 0.01), 
...     'mass_2': 1.3 * (1 + 0.01), 
...     'luminosity_distance': 400,
...     'theta_jn': 5/6 * np.pi,
...     'ra': 3.45,
...     'dec': -0.41,
...     'psi': 1.6,
...     'phase': 0,
...     'geocent_time': 1187008882, 
... }
>>> parameters = pd.DataFrame.from_dict({k:v*np.array([1.]) for k, v in param_dict.items()})

```

<!-- TODO remove redshift! -->

where the values are approximately the [best fit estimates for GW170817](https://doi.org/10.1103/PhysRevX.9.011001), except for the distance, which has been decupled in order to make it a more plausible estimate for a typical neutron star binary.

```{admonition} Why do we multiply by an array?

```{collapse} Click to expand

The reason we are multiplying everything by the array `one` is that the code requires the dataframe to be a mapping between strings and arrays, i.e. for each parameter, its value for any of a number of signals in a population. 
The usage here is to assume that the population consists of one signal.
```

## Simulation

Once we have this population dictionary, we can compute the required errors by defining a [`Network`](#networks) object and using the [`compute_network_errors`](#fisher-matrix-computation) function:

```python
>>> from GWFish.modules.detection import Network
>>> from GWFish.modules.fishermatrix import compute_network_errors
    
>>> network = Network(['ET', 'CE1', 'CE2'])

>>> detected, snr, errors, sky_localization = compute_network_errors(
...    network, 
...    parameters, 
...    waveform_model='IMRPhenomD_NRTidalv2'
... )

>>> print(f'{snr[0]:.0f}')
285

>>> print(f'{errors[0].shape}')
(9,)

```

We are using the `compute_network_errors` function in its simplest form:

- we are omitting to explicitly list the `fisher_parameters`, therefore the function
    is defaulting to including all of the ones which characterize the signal 
    in the Fisher matrix analysis --- which is fine in this case, but may create problems
    in case of perfect degeneracies, such as between redshift and source frame mass;
- we are not simulating the **duty factor** of the detector, since that would mean that 
    our {term}`SNR` is stochastically set to zero (which wouldn't make sense for this example);

We are choosing a waveform approximant that can model the basic features of a neutron
star merger, namely tidal effects. For more details on waveform modelling, see the 
[waveform approximant reference](../how-to/choosing_an_approximant.md).

We are assuming the signal is observed by a network of Einstein Telescope
and two Cosmic Explorers.
The full list of available options, as well as more details on these detectors,
can be found in the [detectors reference](../reference/detectors.md).
If you want to add a new detector, see {ref}`here <how-to/adding_new_detectors:How to add a new detector to GWFish>`.
Finally, our signal is way above the detection threshold for all detectors
involved, but if this was not the case it would not be analyzed. This
is also discussed in the [networks reference](#networks).

## Results

The main result from the computation in the previous section are the Fisher matrix error estimates. For some mathematical details on how those are defined, see the [explanation](../explanation/fisher_matrix.md).

The `errors` array contains the one-sigma errors for all the parameters included in the analysis in order. 


```python
>>> for name, error in zip(parameters.keys(), errors[0]):
...     print(f'{name}: {error:.2e}') 
mass_1: 2.23e-03
mass_2: 2.04e-03
luminosity_distance: 6.19e+01
theta_jn: 2.69e-01
ra: 9.66e-03
dec: 4.71e-03
psi: 6.89e-01
phase: 1.39e+00
geocent_time: 2.87e-05

```

So, for example, the error in distance is `61.9`, in the same units as the distance parameter: the estimated error is $\sigma_{d_L} \approx 61.9 \text{Mpc}$.

The sky localization error is given separately: 

```python
>>> from GWFish.modules.fishermatrix import sky_localization_percentile_factor
>>> print(f'{sky_localization[0]:.2e}')
6.56e-05
>>> print(f'{sky_localization[0] * sky_localization_percentile_factor():.2e}')
9.91e-01

```

The default output from GWFish is a one-sigma error expressed in square radians, 
which is the natural result of the Fisher matrix analysis. 
The convention in astronomy, instead, is to measure these in square degrees (which means we need a conversion factor of $( 180 \text{deg} / \pi \text{rad})^2$) and in 
terms of the $90\%$ sky area. 
This computation is common enough for these analysis that GWFish includes a [convenience function](#utility-functions) for it, labelled `sky_localization_percentile_factor`.

With the conversion done, we can say that the $90\%$ sky area for this (very loud
signal) is expected to be about $\Delta \Omega _{90\%} \approx 1.02 \text{deg}^2$.


```{note}
All errors are given at $1 \sigma$; for any single parameter
this means roughly $68\%$ of the probability mass is expeced to be contained within the 
$1\sigma$ interval, but this number is only $39\%$ for the sky localization, since it 
corresponds to a bivariate distribution
(see, for example, [wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Geometric_interpretation) on this).
In order to convert this figure to a 90% interval, the scaling factor
is given by the inverse survival function of the  
$\chi^2$ distribution with two degrees of freedom evaluated at 10%:
4.605.
This conversion factor (and others like it) can be computed with, 
for example, the following `scipy` code, or the analytical expression:

```python
>>> from scipy.stats import chi2
>>> print(f'{chi2.isf(1 - 0.90, df=2):.3f}')
4.605
>>> print(f'{-2*np.log(1 - 0.90):.3f}')
4.605

```
