import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from processing.preprocessing import normalize
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
import numpy as np
import cv2


def mask_watershed(mask, min_dist = 100, struct=np.array([[0,1,0],[1,1,1],[0,1,0]])):
    """
    Apply a watershed segmentation algorithm to a binary mask image to identify and label distinct objects.

    Parameters
    ----------
    mask : ndarray
        Binary mask image to segment.
    min_dist : int, optional
        Minimum distance between local maxima in the distance transform of the mask. Default is 100.
    struct : ndarray, optional
        Structuring element used to identify local maxima in the distance transform. Default is a 3x3 array with
        a cross-shaped structuring element.

    Returns
    -------
    labels : ndarray
        Labeled image obtained from the watershed segmentation of the input mask.
    """
    m = np.squeeze(mask)
    dt = ndi.distance_transform_edt(m)
    dt[dt<min_dist] = 0
    # localMax = peak_local_max(dt, indices=False, min_distance=min_dist, labels=m)
    localMax = h_maxima(dt, min_dist)  # Normally, this function also incorporates peak_local_max
    markers = ndi.label(localMax, structure=struct)[0]
    labels = watershed(-dt, markers, mask=m)
    return labels


def filter_adjacent_labels(labels, max_adjacent=5, overlap_dist = 1) -> np.ndarray:
    """
    Filters adjacent labels in a label image. Intent is to reduce oversegmentation.

    Parameters:
    -----------
    labels : np.ndarray
        The labeled image to filter. Must be a 2D array of integers, with different objects labeled with different integer
        values greater than 0.
    max_adjacent : int, optional
        The maximum number of adjacent pixels allowed between objects. Objects with more than `max_adjacent` pixels in
        connection with another object will be merged into a single object. Default is 5.

    Returns:
    --------
    np.ndarray
        A new labeled image with adjacent objects merged into a single object. Input dimensions are retained.
    """
    ul = np.unique(labels[labels>0])
    to_skip = []
    output = labels.copy()
    for u in ul:
        if u in to_skip:
            continue
        m = labels.copy()
        im = np.zeros(m.shape)
        im[m==u] = 1

        m[m==u] = 0
        im = ndi.binary_dilation(im, iterations=overlap_dist) - im
        m[im!=1] = 0
        uvs, counts = np.unique(m, return_counts=True)
        for i in range(len(uvs)):
            if i == 0:
                continue
            if counts[i] > max_adjacent:
                output[output==uvs[i]] = u
                to_skip.append(uvs[i])
    # make label values continuous again
    for x, y in enumerate(np.unique(output)):
        output[output==y] = x
    return output


def crop(im, make_square=True):
    """
    Crop the input image to the smallest rectangular area that contains all non-zero pixels.

    Args:
        im: numpy array, the input image.
        make_square: boolean, whether to make the cropped image square by adding zeros to the shorter dimension.

    Returns:
        cropped: numpy array, the cropped image.
    """
    mask = np.zeros(im.shape[:2], dtype=np.uint8)
    if len(im.shape) == 3:
        mask[(np.nanmean(im, axis=2) != 0) & ~(np.isnan(np.nanmean(im, axis=2)))] = 1
    else:
        mask[(im != 0) & ~(np.isnan(im))] = 1

    x, y, w, h = cv2.boundingRect(mask)
    if len(im.shape) == 2:
        cropped = im[y:y + h, x:x + w]
    else:
        cropped = im[y:y + h, x:x + w, :]
    if make_square:
        shape = cropped.shape
        if np.argmin(shape[:2]) == 0:
            if len(shape) == 2:
                cropped = np.vstack([cropped, np.zeros((shape[1]-shape[0], shape[1]), dtype=np.uint8)])
            else:
                cropped = np.vstack([cropped, np.zeros((shape[1]-shape[0], shape[1], shape[2]), dtype=np.uint8)])
        else:
            if len(shape) == 2:
                cropped = np.hstack([cropped, np.zeros((shape[0], shape[0]-shape[1]), dtype=np.uint8)])
            else:
                cropped = np.hstack([cropped, np.zeros((shape[0], shape[0]-shape[1], shape[2]), dtype=np.uint8)])
    return cropped


def fill_holes(mask):
    """
    Fill the holes inside the foreground objects in a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The binary mask of the image, where foreground pixels are represented by
        non-zero values and background pixels are represented by zero values.

    Returns
    -------
    filled_mask : numpy.ndarray
        A binary mask of the same shape as the input mask, where the holes inside
        the foreground objects have been filled.
    """
    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Fill holes in each contour
    for i, contour in enumerate(contours):
        # Create a mask for the contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, i, 255, cv2.FILLED)

        # Update the original mask with the filled contour mask
        filled_mask[contour_mask==255] = 1
    return filled_mask


def kmeans_segmentation(img, sort_idx=2, n_clusters=2, centers=None):
    """
    Uses KMeans clustering to generate an image mask. Works with images with 2+ channels.

    :param img: numpy.ndarray
    Dimensions should be X x Y x n_channels.
    :param sort_idx: int
    K-means clusters are sorted along their average value in a certain channel. For instance,
    on a HSV image, if sort_idx=1, label 0 will have the lowest saturation value, label 1 the second lowest, and so on.
    :param n_clusters: int
    Amount of clusters. 2 generates a binary mask.
    :return: numpy.ndarray of dimensions X x Y. dtype is np.uint8.
    """
    dims = img.shape
    pts = np.reshape(img, (dims[0]*dims[1], dims[2]))

    if centers is None:
        print("KMeans clustering in progress...")
        km = KMeans(init='random', n_clusters=n_clusters)
        km.fit(pts)
        labels = km.labels_.reshape(dims[:-1])

    else:
        print("Using preset centers for KMeans...")
        km = KMeans(init=centers, n_clusters=n_clusters, n_init=1)
        km.fit(centers)
        labels = km.predict(pts).reshape(dims[:-1])

    idx = np.argsort(km.cluster_centers_[:, sort_idx])
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(n_clusters)

    output = np.zeros(dims[:-1])
    for i in range(n_clusters):
        output[labels==i] = lut[i]
    return output.astype(np.uint8), km.cluster_centers_


def dilate_labels(labels, iterations=5):
    """
    Iteratively dilates each label in a label image, for example acquired using watershed segmentation. A label image
    is a 2-D array, preferably np.uint8 dtype, in which each pixel value represents a different label.

    :param labels: np.ndarray of dimensions X by Y. Dtype should be some kind of integer.
    :param iterations: Number of dilation steps to perform.
    :return: np.ndarray of the same dimensions as input image. Label IDs correspond to those of input image.
    """
    out = np.zeros_like(labels, dtype=np.uint8)

    for l in np.unique(labels):
        if l != 0:
            m = np.zeros_like(labels, dtype=np.uint8)
            m[labels==l] = 1
            m = ndi.binary_dilation(m, iterations=iterations)
            out[m==1] = l
    return out


def find_blob_centers(m, to_label=None, color=(255,0,0)):
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    if to_label is None:
        out = np.dstack(3*[255*normalize(m)])
    else:
        out = to_label.copy()
    for i, contour in enumerate(contours):
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            centers.append(center)

            # Draw index label on the center of the blob
            cv2.circle(out, center, 2, color, 2)
            cv2.putText(out, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        except ZeroDivisionError:
            continue

    return out, centers
