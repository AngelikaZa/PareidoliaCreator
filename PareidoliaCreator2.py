""""
Creates random pareidolic images from a set of brushes
Author: A. Zarkali
Date: 3rd October 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product, count
from matplotlib.colors import LinearSegmentedColormap
import os, os.path
import cv2

# parameters
size = 1500
num_desired = 40
directory = r"C:\Users\Angelika\Dropbox\PhD\EXPERIMENTS\01_PriorsAmbiguousStimul\STIMULI\Pareidolia_New\PerlinImages"
out_directory = r"C:\Users\Angelika\Dropbox\PhD\EXPERIMENTS\01_PriorsAmbiguousStimul\STIMULI\Pareidolia_New\Pareidolia"

# generate vectors
def generate_unit_vectors(n):
    'Generates matrix NxN of unit length vectors'
    phi = np.random.uniform(0, 2*np.pi, (n, n))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v

# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t

def generate_2D_perlin_noise(size, ns):
    # generate_2D_array of specific size filled with Perlin noise
    # ns = distance between nodes
    nc = int(size / ns)  # number of nodes
    grid_size = int(size / ns + 1)  # number of points in grid
    # generate grid of vectors
    v = generate_unit_vectors(grid_size)
    # generate some constans in advance
    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)
    # vectors from each of the 4 nearest nodes to a point in the NSxNS patch
    vd = np.zeros((ns, ns, 4, 1, 2))
    for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
        vd[:, :, c, 0] = np.stack(np.meshgrid(l2, l1, indexing='xy'), axis=2)
    # interpolation coefficients
    d = qz(np.stack((np.zeros((ns, ns, 2)),
                     np.stack(np.meshgrid(ad, ad, indexing='ij'), axis=2)),
           axis=2) / ns)
    d[:, :, 0] = 1 - d[:, :, 1]
    # make copy and reshape for convenience
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)
    # make an empy matrix
    m = np.zeros((size, size))
    # reshape for convenience
    t = m.reshape(nc, ns, nc, ns)
    # calculate values for a NSxNS patch at a time
    for i, j in product(np.arange(nc), repeat=2):  # loop through the grid
        # get four node vectors
        av = v[i:i+2, j:j+2].reshape(4, 2, 1)
        # 'vector from node to point' dot 'node vector'
        at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
        # horizontal and vertical interpolation
        t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)
    return m

# generate random perlin noises
for n in range(0, num_desired):
    img1 = generate_2D_perlin_noise(size, 250)
    img2 = generate_2D_perlin_noise(size, 100)
    img3 = generate_2D_perlin_noise(size, 50)
    img4 = generate_2D_perlin_noise(size, 10)
    img5 = generate_2D_perlin_noise(size, 5)
    img6 = generate_2D_perlin_noise(size, 500)
    # combine itterations
    img = (img1 + img2 + img3 + img4 + img5 +img6) / 6
    cmap = LinearSegmentedColormap.from_list('sky',
                                             [(0, '#0572D1'),
                                              (0.75, '#E5E8EF'),
                                              (1, '#FCFCFC')])
    img = cm.ScalarMappable(cmap=cmap).to_rgba(img)
    file_name = os.path.join(directory + "\perlin" + str(n) + ".jpg")
    plt.imsave(file_name, img6)

# apply threshold to images
templateImgs = []
files = os.listdir(directory)
for f in range(len(files)):
    templateImgs.append(os.path.join(directory, files[f]))

def brightness_augment(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv[:, :, 2] += factor
    rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return rgb

# def noisy(image):
#     row,col,ch = image.shape
#     gauss = np.random.randn(row,col,ch)
#     gauss = gauss.reshape(row,col,ch)
#     noisy = image + image * gauss
#     return noisy

kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
def thresholdPerlin(path):
    image = cv2.imread(path)
    image = cv2.filter2D(image, -1, kernel_sharpening)
    image = brightness_augment(image, 100)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.erode(image, (9,9), iterations=5)
    # image = cv2.GaussianBlur(image,(5,5),5)
    retval, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

for i in range(len(templateImgs)):
     image = thresholdPerlin(templateImgs[i])
     #image = cv2.resize(image, (500,500))
     out_name = os.path.join(out_directory + "\pareidolia" + str(i) + ".jpg" )
     cv2.imwrite(out_name, image)
