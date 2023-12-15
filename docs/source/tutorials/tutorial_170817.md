# Estimating SNR and errors for a single signal

This tutorial will show how to compute the SNR and Fisher matrix errors for a 
compact object signal with known parameters.
For more details on the mathematics of how the errors are obtained, see the 
[Fisher matrix reference](../explanation/fisher_matrix.md).

## What we want to do

We will study a GW170817-like signal, as would be seen by LGWA plus Einstein Telescope.
The Fisher matrix approach can give us estimates of the overall SNR it would have, 
as well as the errors in the estimates of the signal parameters.

## Population file

The outcome of this section is a [Pandas `DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing several columns, each corresponding
to one of the parameters characterizing a CBC signal.
The units and conventions for these parameters are outlined in the [reference](../reference/parameters_units.md).

Assuming we already know what the values of the parameters should be, we can generate this dataframe as follows:

```python
>>> import pandas as pd
>>> import numpy as np

>>> param_dict = {
...     'mass_1': 1.4, 
...     'mass_2': 1.3, 
...     'redshift': 0.01,
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

where the values are approximately the [best fit estimates for GW170817](https://doi.org/10.1103/PhysRevX.9.011001), except for the distance, which has been decupled.

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
    
>>> network = Network(['ET', 'CE1', 'CE2'], detection_SNR=(0., 1.))

>>> snr, errors, sky_localization = compute_network_errors(
...    network, 
...    parameters, 
...    waveform_model='IMRPhenomD_NRTidalv2'
... )

>>> print(f'{snr[0]:.0f}')
284

>>> print(f'{errors.shape}')
(10,)

```

We are using the `compute_network_errors` function in its simplest form, 

<!-- the value used to be 742 - which is correct? -->
<!-- TODO do not use equal mass as an example -->
<!-- TODO mention duty factor -->
<!-- TODO link to waveform approximant section -->
<!-- TODO explain detection_SNR -->

{ref}`reference page <reference/API:Fisher Matrix Computation>`.

```{admonition} Which detectors are available?
:class: seealso

The full list of available options, as well as more details on these detectors,
can be found in the [detectors reference](../reference/detectors.md).
If you want to add a new detector, see {ref}`here <how-to/adding_new_detectors:How to add a new detector to GWFish>`.
```


## Results

The `errors` array contains the one-sigma errors for all the parameters included in the analysis in order. 

```python
>>> for name, error in zip(parameters.keys(), errors[0]):
...     print(f'{name}: {error:.2e}') 
mass_1: 5.93e-04
mass_2: 5.51e-04
redshift: 3.08e-06
luminosity_distance: 6.39e+01
theta_jn: 2.78e-01
ra: 9.76e-03
dec: 4.71e-03
psi: 7.17e-01
phase: 1.44e+00
geocent_time: 2.72e-05
```

So, for example, the second-to-last value is `3.962E-01`, corresponding to the error in the phase:
the estimated error is $\sigma_\varphi \approx 0.3962 \text{rad}$.

The last value is `2.264E-05` for the sky localization: 
$\Delta \Omega \approx 2.264 \times 10^{-5} \text{sr} \approx 0.07 \text{deg}^2$ 
(the conversion factor is $(180 / \pi)^2$).

```{caution}
All errors are given at $1 \sigma$; for any single parameter
this means roughly $68\%$ of the probability mass is expeced to be contained within the 
$1\sigma$ interval, but this number is only $39\%$ for the sky localization, since it 
corresponds to a bivariate distribution
(see, for example, [wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Geometric_interpretation) on this).
In order to convert this figure to a 90% interval the conversion factor
is given by the inverse survival function of the  
$\chi^2$ distribution with two degrees of freedom evaluated at 10%:
4.605, so in this case the 90% area would be roughly $0.07 \text{deg}^2\times 4.6 \approx 0.3 \text{deg}^2$.
This conversion factor (and others like it) can be computed with, 
for example, the following `scipy` code, or the analytical expression:

```python
>>> from scipy.stats import chi2
>>> print(f'{chi2.isf(1 - 0.90, df=2):.3f}')
4.605
>>> print(f'{-2*np.log(1 - 0.90)}:.3f')
4.605
```

This computation is common enough for these analysis that GWFish includes a [convenience function](#utility-functions) for it, labelled `sky_localization_percentile_factor`.
```