# Estimating SNR and errors for a single signal

This tutorial will show how to compute the SNR and Fisher matrix errors for a 
compact object signal with known parameters.

## What we want to do

We will study a GW170817-like signal, as would be seen by LGWA plus Einstein Telescope.
The Fisher matrix approach can give us estimates of the overall SNR it would have, 
as well as the errors in the estimates of the signal parameters.

Since GWFish is geared towards giving estimates for whole populations of signals as opposed to
single ones, there are two steps to this process: 

- generating a `hdf5` file containing the parameters of all the binaries at hand ,
    which is typically aided by the `CBC_Population.py` script, but which we will accomplish
    with a simpler custom script;
- computing the associated SNRs and Fisher matrix errors,
    which can be accomplished with the `CBC_Simulation.py` script.

## Population file

The outcome of this section is an `hdf5` file containing several columns, each corresponding
to one of the parameters characterizing a CBC signal.

- `'mass_1'`: the mass of the primary object, in solar masses;
- `'mass_2'`: the mass of the secondary object, in solar masses;
- `'redshift'`: the redshift of the merger;
- `'luminosity_distance'`: the luminosity distance of the merger, in Megaparsec;
- `'theta_jn'`: the inclination angle (between the observation direction 
    and the angular momentum of the merger), in radians;
- `'psi'`: the polarization angle of the merger, in radians;
- `'phase'`: the initial phase of the waveform, in radians;
- `'ra'`: the right ascension of the merger, in radians;
- `'dec'`: the declination of the merger, in radians;
- `'geocent_time'`: the time of merger, expressed as [GPS time](https://www.andrews.edu/~tzs/timeconv/timeconvert.php?)

Now, the code required to write this file is quite simple if 
we alredy have access to the values of the parameters: it will be a script in the form

```python
#!/usr/bin/env python3
import pandas as pd
import numpy as np

one = np.array([1])
parameters = pd.DataFrame.from_dict({
    'mass_1': 1.4 * one, 
    'mass_2': 1.4 * one, 
    'redshift': 0.01 * one,
    'luminosity_distance': 40 * one,
    'theta_jn': 5/6 * np.pi * one,
    'ra': 3.45 * one,
    'dec': -0.41 * one,
    'psi': 1.6 * one,
    'phase': 0 * one,
    'geocent_time': 1187008882 * one, 
})
parameters.to_hdf('170817_like_population.hdf5', mode='w', key='root')
```

where the values are approximately the [best fit estimates for GW170817](https://doi.org/10.1103/PhysRevX.9.011001).

```{admonition} Why is the script like this?

```{collapse} Click to expand

The reason we are multiplying everything by the array `one` is that the 
{meth}`from_dict <pandas.DataFrame.from_dict>` classmethod requires data in the form of a dictionary like `{field_name: array-like}`; what we are effectively doing is creating a matrix with one row and many columns.

Passing through `pandas` is not strictly necessary but it's quite convenient, 
since it is able to easily convert its native `DataFrame` object into the `hdf5` 
files `GWFish` is able to read.

```

Take the previous code block and save it as a script such as `population.py`; 
then, run it with 

```bash
python population.py
```

## Simulation

Once we have this population file, we can simply run `CBC_Simulation.py` to 
compute the required errors:

```bash
python CBC_Simulation.py --pop_id 170817like --pop_file 170817_like_population.hdf5 --detectors ET LGWA --networks "[[0, 1]]"
```

The output should look similar to

```bash
LAL package is not installed. Only GWFish waveforms available.
Processing CBC population
<...>
Network: ET_LGWA
Detected signals with SNR>9.000: 1.000 (1 out of 1); z<0.010
SNR: 720.575 (min) , 720.575 (max) 
--- 1.3764996528625488 seconds ---
```

```{admonition} Which detectors are available?
:class: seealso

The full list of available options, as well as more details on these detectors, 
can be found in `GWFish/detectors.yaml`.
```

## Results

Running the `CBC_Simulation.py` script will generate two files:
`Signals_170817like.txt` and `Errors_ET_LGWA_170817like_SNR9.0.txt`.

The first is simply a recap of the parameters we used, plus the computed SNR(s).
It should look like:
```
# mass_1 mass_2 redshift luminosity_distance theta_jn ra dec psi phase geocent_time ET_LGWA_SNR
1.400 1.400 0.010 40.000 2.618 3.450 -0.410 1.600 0.000 1187008882.000 720.575
```

### Reading the errors file

The Fisher matrix errors will be saved to a file named 
`Errors_ET_LGWA_170817like_SNR9.0.txt`, containing

- a copy of the input parameters, with the same labels as before; 
- the resulting SNRs, labelled `network_SNR`;
- the corresponding Fisher-matrix errors, with columns labelled `err_<param>` for every parameter;
- additionally, the sky localization area in steradians, labelled `err_sky_location`.

```{note}
The reason for the `SNR9.0` label is that a cut is performed at that (customizable)
SNR: signals below the threshold are not expected to be statistically distinguishable from noise,
so they should not be included in the simulation.

In our case, though, the source is so close that even in the worst case scenario
the SNR is still well above the threshold. 
```

It should look like:

```
network_SNR mass_1 mass_2 redshift luminosity_distance theta_jn ra dec psi phase geocent_time err_ra err_dec err_psi err_theta_jn err_luminosity_distance err_mass_1 err_mass_2 err_geocent_time err_phase err_sky_location
720.5753783784745 1.400E+00 1.400E+00 1.000E-02 4.000E+01 2.618E+00 3.450E+00 -4.100E-01 1.600E+00 0.000E+00 1.187E+09 3.053E-03 2.591E-03 1.976E-01 1.011E-01 2.264E+00 4.372E-08 4.372E-08 5.555E-05 3.962E-01 2.264E-05 
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
```