import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import gaussian_filter

img = np.asarray(image.imread("580data.png")) # input image. newData.png = 37x37. 580data = 147x147

# Standard Deviation

for i in np.arange(0, 5.1, 0.2): # range of std values
    i = round(i, 1)
    gaussian = gaussian_filter(img, i)
    fig = plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(img)
    ax2.imshow(gaussian)
    filename = "TestDataNew580/std/std" + str(i) + ".png"
    plt.savefig(filename)
    plt.close


# Contrast

for i in np.arange(0.0, 5.1, 0.2):
    i = round(i, 1)
    for j in np.arange(0.8, 3.1, 0.2): # range of contrast values for each std value
        j = round(j, 1)
        imgstd = gaussian_filter(img, i)
        con = imgstd * j
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(imgstd)
        ax2.imshow(con)
        filenameClose = "TestDataNew580/contrast/std" + str(i) + "con" + str(j) + ".png"
        plt.savefig(filenameClose)
        plt.close()


# Brightness

for i in np.arange(0.0, 5.1, 0.2):
    i = round(i, 1)
    for j in range(-50, 51, 10): # range of brightness values for each std value
        imgstd = gaussian_filter(img, i)
        bri = imgstd + j
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(imgstd)
        ax2.imshow(bri)
        filenameClose = "TestDataNew580/brightness/std" + str(i) + "bri" + str(j) + ".png"
        plt.savefig(filenameClose)
        plt.close()

