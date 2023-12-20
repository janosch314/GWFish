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
the __[`GWFish` paper](https://www.sciencedirect.com/science/article/abs/pii/S2213133722000853?via%3Dihub)__.

This software is developed by the [gravitation group](https://wikiet.gssi.it/index.php/Main_Page) 
at the [Gran Sasso Science Institute](https://www.gssi.it/).

```{seealso}
This documentation is written according to the [diátaxis framework](https://diataxis.fr).
```

```{toctree}
:glob:
:maxdepth: 1
:titlesonly:
:caption: Introduction

installation.md
glossary.md
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
:caption: Explanation

explanation/*
```

```{toctree}
:glob:
:maxdepth: 1
:titlesonly:
:caption: Technical Reference

reference/*
```

