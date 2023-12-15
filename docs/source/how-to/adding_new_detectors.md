# How to add a new detector to GWFish

Adding a new detector to GWFish is relatively straightforward, as long as it
is in the same class as an existing one. 
Let us suppose the detector you want to add falls into one of the 
{ref}`existing categories <reference/detectors:Detector properties>`.

Modify the `GWFish/detectors.yaml` file, adding an entry 
for the new detector's parameters, and if needed add its PSD to 
the `GWFish/detector_psd` folder. 
Refer to the {ref}`detector documentation <reference/detectors:Detector properties>` for the 
conventions on how to format these parameters.

Remember to give the new detector a unique identifier.
Now, it will be possible to refer to this detector when generating a [`Detector` of `Network` object](../reference/detectors.md).
