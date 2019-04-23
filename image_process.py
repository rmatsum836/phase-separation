import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, convolve1d
from skimage.io import imread, imsave
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


def apply_otsu(gray,dom_color,out_file=None, save=True):
    thresh_otsu = threshold_otsu(gray)
    if dom_color == 2:
        im_bw = gray > thresh_otsu
    elif dom_color == 0:
        im_bw - gray < thresh_otsu

    if save == True:
        imsave(out_file, gray)

    return im_bw


def remove_border(image):
    image[:,:90] = 0
    image[:,-90:] = 0
    image[:90,:] = 0
    image[-90:,:] = 0

    return image


def get_dominant_color(image, clusters=1):
    dc = DominantColors(image, clusters)
    colors = dc.dominantColors()
    dominant = list(colors[0]).index(max(colors[0]))

    return dominant


def cutoff_particles(image, image_props, cutoff=300, out_file=None, save=None):
    im_bw_filt = image > 1

    n_regions = 0
    for prop in im_props:
        if prop.area < cutoff:
            im_bw_filt[image==prop.label] == False
        else:
            n_regions += 1

    if save == True:
        imsave(out_file, im_bw_filt)

    print('Number of individual regions = {}'.format(n_regions))

    return n_regions


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters, filename):
        self.CLUSTERS = clusters
        self.IMAGE = image
        self.FILE = filename

    def dominantColors(self):

        #read image
        img = cv2.imread(self.IMAGE)
        img = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)

        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        #save image after operations
        self.IMAGE = img

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)


        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        #save labels
        self.LABELS = kmeans.labels_


        #returning after converting to integer from float
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2])) 
    def plotClusters(self):
        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        ax.set_xlabel('Red', fontsize=18, labelpad=13)
        ax.set_ylabel('Green', fontsize=18, labelpad=13)
        ax.set_zlabel('Blue', fontsize=18, labelpad=16)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        plt.tight_layout()
        plt.savefig(self.FILE)
        plt.show()
