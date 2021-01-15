# Import libraries
from copy import deepcopy
import cv2
import numpy as np
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese, checkerboard_level_set)
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# calculation of chan vese segmentation
def calculate_chanVese(img):
    gray = img.sum(-1)
    image = img_as_float(gray)
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)
    print("chan vese done")

    return ls


# calculation of canny edges segmentation
def calculate_canny_edges(img):
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2, 2), np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    for i in range(0):
        closing = cv2.morphologyEx(sure_bg, cv2.MORPH_CLOSE, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(closing, kernel, iterations=3)

    edged = cv2.Canny(sure_bg, 30, 300)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ret2 = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    print("canny edges done")

    return ret2


# calculation of kmeans segmentation
def calculate_kmeans(img):
    x = np.array(img)

    z = np.dstack((x, rgb2hsv(x)))
    starting_shape = z.shape

    vectorized = np.float32(z.reshape((-1, 6)))

    kmeans = KMeans(random_state=0, init='random', n_clusters=4)
    labels = kmeans.fit_predict(vectorized)

    pic = labels.reshape(starting_shape[0], starting_shape[1])
    print("kmeans done")

    return pic


# present the segmented images on a plot
def present_plot(img, canny, k, chan, ls):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    ax = axes.flatten()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(canny, cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("Canny Edges Segmentation", fontsize=12)

    ax[2].imshow(chan, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Chan Vese Snakes Unsupervised Segmentation", fontsize=12)

    ax[3].imshow(k)
    ax[3].set_axis_off()
    ax[3].set_title("Kmeans Segmentation", fontsize=12)

    plt.show()


if __name__ == "__main__":
    img = cv2.imread('image_for_segmentation.jpg')

    canny_edges = deepcopy(img)
    chanVese = deepcopy(img)
    kmeans = deepcopy(img)
    # call the functions
    canny_edges = calculate_canny_edges(canny_edges)
    ls = calculate_chanVese(chanVese)
    kmeans = calculate_kmeans(kmeans)
    # present the results
    present_plot(img, canny_edges, kmeans, chanVese, ls)
