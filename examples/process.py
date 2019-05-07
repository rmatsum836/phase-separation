from skimage.io import imread, imsave
from scipy.ndimage import gaussian_filter
import image_process
import numpy as np
import glob
import os

cwd = os.getcwd()
for filepath in glob.iglob('{}/data/train-data/hetero/*'.format(cwd)):
    rindex = -5
    sindex = 3 
    out_file = filepath[0:rindex] + "otsu-" + filepath[rindex:]
    out_file = filepath[0:sindex] + '-otsu' + filepath[sindex:]
    hetero_list = list()
    unsharp_strength = 0.8
    kernel_size = 8
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size
    kernel[0,:]
    image = imread(filepath)
    blurred = gaussian_filter(image, sigma=8)
    convolved = image_process.convolveImage(image - unsharp_strength * blurred, kernel)
    gray = convolved[:,:,0]
    im_bw = image_process.apply_otsu(gray, out_file=out_file, save=True)
