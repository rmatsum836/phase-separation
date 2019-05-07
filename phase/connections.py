import os
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
from skimage.color import label2rgb
import glob


def label_regions(image, filename, plot=False):
    total_area = list()
    label_image = label(image)
    image_label_overlay = label2rgb(label_image, image=image)

    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 1000:
            total_area.append(region.area)
            # draw rectangle around segmented coins
            if plot == True:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

    if plot == True:
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(filename)

    return np.sum(total_area)


def _cutoff_particles(image, image_props, cutoff=300):
    im_bw_filt = image > 1

    # Loop through image properties and delete small objects
    n_regions = 0
    for prop in image_props:
        if prop.area < cutoff:
            im_bw_filt[image==prop.label] == False
        else:
            n_regions += 1

    print('Number of individual regions = {}'.format(n_regions))
    
    return n_regions


def count_connections(filepath, labeled_path, img_filetype='png', plot=False):
    component_list = list()
    max_list = list()
    total_list = list()
    for img_file in glob.iglob('{}/*.{}'.format(filepath, img_filetype)):
        image = imread(img_file)
        filename = filepath.split('/')[-1]
        test = label_regions(image, '{}/{}'.format(labeled_path,filename), plot=True)
        im_labeled, n_labels = label(image, background=0, return_num=True)
        im_labeled += 1
        im_props = regionprops(im_labeled)
        n_regions = _cutoff_particles(im_labeled, im_props, cutoff=1000)
        if len(regionprops(label(image))) == 0:
            max_area = 0
        else:
            max_area = np.max([region.area for region in regionprops(label(image))])

        component_list.append(n_regions)
        total_list.append(test)
        max_list.append(max_area)

    return component_list, total_list, max_list
