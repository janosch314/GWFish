# How to add a new detector to GWFish

Adding a new detector to GWFish is relatively straightforward, as long as it
is in the same class as an existing one. 

The detector class can currently be one of the following:

- `earthDelta` (e.g. Einstein Telescope)
- `earthL` (e.g. LIGO, Virgo, Cosmic Explorer)
- `satellitesolarorbit` (e.g. LISA)
- `lunararray` (e.g. LGWA)

Let us suppose the detector we want to add falls into one of these categories.
The way to add it, then, is to modify the `GWFish/detectors.yaml` file.

```{todo}
This how-to guide is unfinished.
```