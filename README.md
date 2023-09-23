# Drone_image_classification
Landcover classification of drone UAV imagery (RGB and Multispectral) using Random forest with unsupervised segmentation (superpixel) postprocessing.

## Description

In this repository I only store the code. The data is taken from [Hilden](https://arcticdrones.org/) collection, which is accessible by contacting the Hilden Network.

The code is organized into 4 Jupyter notebooks and one Python script with functions:
* 0._ - preprocessing of the images. Filling the gaps and resizing the images if needed.
* 1._ - **showing an example of landcover classification on one image**
* 2._ - classifying all the prepared in script 0. multispectral imagery
* 3._ - classifying all the prepared in script 0. RGB imagery
* myfunctions - all the custom functions I used

## Visualization

If you want to quickly glance at how the analysis is done, just view the [1._](https://github.com/PlekhanovaElena/Drone_image_classification/blob/main/1._RF_on_MSP_with_spp_one_image_example.ipynb) script. It contains images and descriptions of each step.

Please feel free to contact me if you have any questions!
