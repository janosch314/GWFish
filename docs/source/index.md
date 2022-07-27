# GWFish

GWFish is a Fisher matrix code geared towards future gravitational-wave detectors.

## Fisher matrix basics

These estimates are obtained by considering a quadratic approximation to the likelihood 
(valid in the high-SNR limit), in the form 

$$ \mathcal{L} \propto \exp \left( - \frac{1}{2} \Delta \theta ^i \Gamma_{ij} \Delta \theta ^j \right)
$$

where $\Delta \theta = \theta - \overline{\theta}$ is the vector of the errors in our
estimates for the parameters, $\overline{\theta}$ being the vector of the true values.
The matrix $\Gamma$ is computed as 

$$\Gamma_{ij} = 
\left( 
    \frac{\partial h}{\partial \theta _i} 
    \left| 
    \frac{\partial h}{\partial \theta _j} 
\right.\right)
$$

where $h$ is the strain at the detector corresponding to the parameters $\theta$,
the product denoted as $(h|g)$ is 

$$ (h | g) = 4 \Re \int_0^{ \infty } \mathrm{d}f \frac{h^* (f) g(f)}{S_n(f)}\,,
$$

$S_n$ being the poower spectral density of the noise.

The covariance matrix is therefore the inverse of $\Gamma _{ij}$, and the variance
of each parameter can be computed as 

$$ \operatorname{var} \theta _i = (\Gamma^{-1})_{ii}
$$

where no summation is intended.
Intuitively, this is reasonable: if the measurable waveform varies little 
in the direction of a specific parameter, that parameter will be hard to constrain.



```{toctree}
:glob:
:maxdepth: 1
:titlesonly:
:caption: Tutorials

tutorials/*
```

```{toctree}
:glob:
:maxdepth: 1
:titlesonly:
:caption: How-to guides

how-to/*
```

```{toctree}
:glob:
:maxdepth: 1
:titlesonly:
:caption: Technical Reference

reference/*
```

This documentation is meant to follow the [diátaxis framework](https://diataxis.fr).
