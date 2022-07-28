# Randomizing the extrinsic parameters

We want to achieve the same result as in the [previous tutorial](tutorial_170817.md),
but this time instead of fixing all the parameter values, we will randomize some of them,
in order to get a better feel of how the error distribution looks in general.

This is still not a realistic scenario --- for that, we'd need to look at the 
mass distribution of neutron star mergers in redshift at the very least ---
but it illustrates several points which are relevant for more advanced usage.

## Randomized population file

We will need to generate some random points on a sphere for the angular distribution
both of the source in the sky, and its orientation with respect to the observation axis.

The point $(\theta, \varphi)$ is uniformly distributed on the sphere if 
$\varphi \sim \mathcal{U}(0, 2 \pi )$ while $\theta \sim \sin \theta $.
Generating both angles with a uniform distribution would bias our points towards the poles.

The easiest way to generate a point distributed like a sine is to use the relation

$$ p(\theta ) = \sin \theta \mathrm{d} \theta = \mathrm{d}\cos \theta 
$$

therefore we can generate a number $x \sim \mathcal{U}(-1, 1)$ and take its arccosine
to get a sine-distributed variable.

We will generate 100 such samples - not really enough to get good statistics, but 
it will suffice for this tutorial, and it will allow us to run quickly.

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng()

ns = 100
one = np.ones((ns,))
parameters = pd.DataFrame.from_dict({
    'mass_1': 1.4*one, 
    'mass_2': 1.4*one, 
    'redshift': 0.01*one,
    'luminosity_distance': 40*one,
    'theta_jn': np.arccos(rng.uniform(-1., 1., size=(ns,))),
    'dec': np.arccos(rng.uniform(-1., 1., size=(ns,))) - np.pi / 2.,
    'ra': rng.uniform(0, 2. * np.pi, size=(ns,)),
    'psi': rng.uniform(0, 2. * np.pi, size=(ns,)),
    'phase': rng.uniform(0, 2. * np.pi, size=(ns,)),
    'geocent_time': rng.uniform(1735257618, 1766793618, size=(ns,)) # full year 2035
})
parameters.to_hdf('170817_like_population.hdf5', mode='w', key='root')
```

This will generate a population file for which we can generate the 
Fisher matrix errors just like discussed in the 
{ref}`simulation section of the previous tutorial <tutorials/tutorial_170817:Simulation>`.

```{note}
This time, it will take on the order of a couple minutes on a typical laptop.
You should see a progressbar on your terminal screen.
```

## Interpreting the results

The results, as before, will be saved in a file named `'Errors_ET_LGWA_170817like_SNR9.0.txt'`.
This is simple to work with if we use `pandas`:
we can open it with 

```python
import pandas as pd
df = pd.read_csv('Errors_ET_LGWA_170817like_SNR9.0.txt', sep=' ')
print(df.keys())
```

which should output:

```
Index(['network_SNR', 'mass_1', 'mass_2', 'redshift', 'luminosity_distance',
       'theta_jn', 'dec', 'ra', 'psi', 'phase', 'geocent_time', 'err_ra',
       'err_dec', 'err_psi', 'err_theta_jn', 'err_luminosity_distance',
       'err_mass_1', 'err_mass_2', 'err_geocent_time', 'err_phase',
       'err_sky_location'],
      dtype='object')
```

These are all columns we have access to, so we can easily make plots. 
The following code snippets are all __self-contained__, and can each be copy-pasted
into a `python` script and executed.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('text', usetex=True)

df = pd.read_csv('Errors_ET_LGWA_170817like_SNR9.0.txt', sep=' ')

skyloc_error = df['err_sky_location'] * (180/np.pi)**2
plt.hist(np.log(skyloc_error), bins=20, density=True)

plt.xlabel('Sky localization error, square degrees')
plt.ylabel('Probability density per e-fold')
plt.gca().xaxis.set_major_formatter(lambda x, pos: f'${np.exp(x):.2g}$')
plt.savefig('sky_localization_histogram.png', dpi=250)
```

```{figure} ../figures/sky_localization_histogram.png

Histogram of the sky-localization of a GW170817-like signal, with randomized orientation.
```

or scatter plots:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('text', usetex=True)

df = pd.read_csv('Errors_ET_LGWA_170817like_SNR9.0.txt', sep=' ')

skyloc_error = df['err_sky_location'] * (180/np.pi)**2
snr = df['network_SNR']

plt.scatter(np.log(skyloc_error), np.log(snr))

plt.xlabel('Sky localization error, square degrees')
plt.gca().xaxis.set_major_formatter(lambda x, pos: f'${np.exp(x):.2g}$')

plt.ylabel('Network SNR')
plt.gca().yaxis.set_major_formatter(lambda x, pos: f'${np.exp(x):.0f}$')
plt.savefig('snr_skyloc_scatter.png', dpi=250)
```

```{figure} ../figures/snr_skyloc_scatter.png

Scatter plot of the sky localization against the signal SNR.
```
