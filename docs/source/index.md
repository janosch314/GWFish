# GWFish

GWFish is a Fisher matrix code geared towards future gravitational-wave detectors.

It is able to provide estimates of the signal-to-noise ratio and of the errors on 
our estimates of the parameters, for a signal as it would be seen by one or more 
future detectors, among:

- [LIGO](https://ligo.caltech.edu) / [Virgo](https://www.virgo-gw.eu/) in their fifth observing run;
- [Kagra](https://gwcenter.icrr.u-tokyo.ac.jp/en/);
- [Einstein Telescope](http://www.et-gw.eu);
- [Lunar Gravitational Wave Antenna](http://socrate.cs.unicam.it/);
- [Cosmic Explorer](https://cosmicexplorer.org);
- [LISA](https://lisamission.org);
- Voyager.

It is able to account for a time-varying antenna pattern, since we expect these
detectors to be sensitive in the low-frequency regime, for which the motion of the 
Earth / Moon / satellites is significant across the signal duration.

For more information about the theory than what is discussed here, refer to 
the __[`GWFish` paper](https://arxiv.org/abs/2205.02499)__ (preprint, submitted to 
[Astronomy and Computing](https://www.journals.elsevier.com/astronomy-and-computing)).

## Fisher matrix basics

The estimates in GWFish are obtained by considering a quadratic approximation to the likelihood 
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

$S_n$ being the power spectral density of the noise.

The covariance matrix is therefore the inverse of $\Gamma _{ij}$, and the variance
of each parameter can be computed as 

$$ \operatorname{var} \theta _i = (\Gamma^{-1})_{ii}
$$

where no summation is intended.
Intuitively, this is reasonable: if the measurable waveform varies little 
in the direction of a specific parameter, that parameter will be hard to constrain.

```{seealso}
This documentation is written according to the [diátaxis framework](https://diataxis.fr).
```

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

