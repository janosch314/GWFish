# Randomizing the extrinsic parameters

We want to achieve the same result as in the [previous tutorial](tutorial_170817.md),
but this time instead of fixing all the parameter values, we will randomize some of them,
in order to get a better feel of how the error distribution looks in general.

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
parameters.to_hdf('170817_like_population', mode='w', key='root')
```

This will generate a population file for which we can generate the 
Fisher matrix errors just like discussed in the 
{ref}`simulation section of the previous tutorial <tutorials/tutorial_170817:Simulation>`.

## Interpreting the results

```{warning} TODO

Make some plots with the results!
```