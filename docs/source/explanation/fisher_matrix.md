# Fisher matrix basics

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

## Computational challenges

One of the most difficult steps in the computation outlined above is the inversion of the 
matrix $\Gamma$, not because of its high dimensionality (it's on the order of $10\times 10$)
but because it is prone to having singular rows/columns.

For example, consider a system seen head-on (observation direction aligned 
with the angular momentum, $\theta _{JN} = 0$). 
Since the waveform (considering only the $\ell = m = 2$ multipole) depends on this angle 
only through $\cos( \theta _{JN})$, the derivative will scale with

$$ \frac{ \mathrm{d} h}{\mathrm{d} \theta _{JN}} 
\propto \frac{ \mathrm{d} \cos( \theta _{JN})}{\mathrm{d} \theta _{JN}} 
= 0,
$$

which means that the whole row and column corresponding to this parameter will vanish. 
Even if the system is not exactly head-on, this will still correspond to 
a row-column of very low numbers, leading to numerical noise and instability.