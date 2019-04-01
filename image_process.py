import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, convolve1d
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion, dilation

"""
A collection of functions to process images of ionic liquid-solvent mixtures
"""

def convolveImage(image, kernel):
    def scaleIt(cvld):
        cvld[cvld > 255.0] = 255.0
        cvld[cvld < 0.0] = 0.0
        return cvld
    convolved = np.ones(image.shape)
    for i in range(convolved.shape[-1]):
        cvld = convolve2d(image[:,:,i], kernel, boundary='fill', mode='same',
                          fillvalue=0.)
        convolved[:,:,i] = scaleIt(cvld)

    return convolved.astype(int)


def apply_otsu(gray):
    thresh_otsu = threshold_otsu(gray)
    im_bw = gray > thresh_otsu

    return im_bw


def cutoff_particles(image, image_props, cutoff=300):
    im_bw_filt = image > 1

    n_regions = 0
    for prop in im_props:
        if prop.area < cutoff:
            im_bw_filt[image==prop.label] == False
        else:
            n_regions += 1

    print('Number of individual regions = {}'.format(n_regions))

    return n_regions
