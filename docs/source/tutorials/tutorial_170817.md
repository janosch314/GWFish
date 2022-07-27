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
parameters.to_hdf('170817_like_population', mode='w', key='root')
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
the basic syntax is 

```bash
python CBC_Simulation.py --pop_id 170817like --pop_file 170817_like_population --detectors ET LGWA --networks "[[0, 1]]"
```

This will use the default network: just Einstein Telescope. 
In order to perform the same analysis for different networks, one may use the 
`--detectors` argument, followed by a list of detectors to use, and then 
the `--networks` argument to choose which combinations to consider.
An example is as follows:

```bash
python CBC_Simulation.py --pop_id 170817like --pop_file 170817_like_population --detectors ET CE1 CE2 --networks [[0], [0, 1], [0, 1, 2]]
```
which means performing the analysis first with only Einstein Telescope,
then with ET as well as one Cosmic Explorer,
then with ET and two Cosmic Explorers at different locations.

The full list of available options, as well as more details on these detectors, 
can be found in `GWFish/detectors.yaml`.

## Results

### Reading the results file

The results from the simulation (in the ET only case) will be saved to a file named 
`Errors_ET_170817like_SNR9.0.txt`, a space-separated text file containing

- a copy of the input parameters, with the same labels as before; 
- the resulting SNRs, labelled `network_SNR`;
- the corresponding Fisher-matrix errors, with columns labelled `err_<param>` for every parameter;
- additionally, the sky localization area in steradians, labelled `err_sky_location`.



### Interpreting the results



All errors are given at $1 \sigma$; note that for any single parameter
this means roughly $68\%$ of the probability mass is expeced to be contained within the 
$1\sigma$ interval, but this number is only $39\%$ for the sky localization, since it is bivariate
(see, for example, [wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Geometric_interpretation) on this).