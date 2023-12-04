import numpy as np
from scipy.ndimage import morphology
import cv2
import matplotlib
import imutils
import os
import glob
from tkinter import filedialog, Tk

matplotlib.use('qt5agg')
from matplotlib import animation
import matplotlib.pyplot as plt

COLORS = {
    'red': [0],
    'green': [1],
    'blue': [2],
    'yellow': [0, 1],
    'magenta': [0, 2],
    'cyan': [1, 2],
    'white': [0, 1, 2],
    'black': []
}

"""
The below functions are all shorthand conversion functions.
"""


def bgr2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def rgb2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def bgr2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def rgb2bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def bgr2hsv(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


def bgr2lab(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2LAB)


def rgb2hsv(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2HSV)


def rgb2lab(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2LAB)


def animate_array(arr, fps=1):
    """
    Animates a 3-D axis along the last axis.
    :param arr:
    :param fps:
    :return:
    """
    fig = plt.figure()
    im = plt.imshow(arr[:, :, 0],
                    interpolation='none',
                    aspect='auto')
    plt.colorbar()

    def animate_func(i):
        im.set_array(arr[:, :, i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        interval=1 / fps,
        repeat=True,
        frames=arr.shape[-1]
    )
    return anim


def mask_to_outline(mask):
    """
    Converts a binary mask to an outline by subtracting it from the dilated mask.
    :param mask: A binary image.
    :return: A binary image.
    """
    md = morphology.binary_dilation(mask, structure=np.ones((3, 3)), iterations=1)
    out = md - mask
    out[out == 0] = np.nan
    return md - mask


def show_image_and_mask(im, mask, ax=None, color='red', alpha=1, outline=True):
    """
    Shows an RGB image with an overlay. Can provide an axis to use in subplots.
    :param im: A RGB image.
    :param mask: A binary mask (float or int).
    :param ax: Optional: axes object to plot to.
    :param color: A string specifying color. Accepted values: ['red', 'green', 'blue', 'yellow',
    'magenta', 'cyan', 'white', 'black']
    :param alpha: A float between 0 and 1. 1 = opaque, 0 = fully transparent.
    :param outline: If True, show only the outline. If False, show the mask as an overlay.
    """
    if outline:
        m = mask_to_outline(mask)
    else:
        m = np.copy(mask)
    m2 = np.zeros(m.shape + (4,))
    m2[:, :, 3] = m * alpha

    for i in range(3):
        if i in COLORS[color]:
            m2[:, :, i] = m

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(im)
    ax.imshow(m2)
    ax.axis('off')


def show_image_and_outline(im, mask, color='red', ax=None, weight=1):
    c = np.zeros(3)
    c[COLORS[color]] = 255
    cont, h = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(cv2.drawContours(np.copy(im), cont, -1, c, weight))


def show_multiple_annotations(img, labels, ax=None, weight=1, annotations=None, fontscale=1):
    col_ids = list(COLORS.keys())
    target = np.float32(img).copy()
    if annotations is None:
        anns = [str(l) for l in np.unique(labels)]
    else:
        anns = annotations

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(labels.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        color_id = label % len(col_ids)
        color = COLORS[col_ids[color_id]]
        target = cv2.drawContours(target, c, -1, color, weight)
        x, y, w, h = cv2.boundingRect(c)
        target = cv2.rectangle(target, (x, y), (x + w, y + h), color, weight)
        target = cv2.putText(target, anns[label], (x, y + h + 12), cv2.FONT_HERSHEY_PLAIN, fontscale, color, 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(target)


def extract_tags_per_label(labels):
    out = []
    for l in np.unique(labels):
        size = np.sum(labels == l) / 1000.0
        out_label = "id:{}/{}k px".format(l, np.round(size, 1))
        out.append(out_label)
    return out


def extract_statistics_per_label(labels, id=None):
    out = []
    for l in np.unique(labels):
        if l == 0:
            continue
        mask = np.zeros(labels.shape)
        mask[labels == l] = 1
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = cv2.arcLength(contours[0], closed=True)
        size = cv2.contourArea(contours[0])
        if len(contours[0]) > 4:
            c, xy, r = cv2.fitEllipse(contours[0])
            elongation = np.max(xy) / np.min(xy)
        else:
            x, y, w, h = cv2.boundingRect(contours[0])
            elongation = w / h

        data = [l, perim, size, elongation]
        if id is not None:
            data = [id] + data
        out.append(data)

    return out


def padding_2d(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def pad_rgb(rgb_im, xx, yy):
    r, g, b = [rgb_im[:, :, c] for c in range(3)]
    out = np.asarray([padding_2d(i, xx, yy) for i in [r, g, b]])
    return np.moveaxis(out, 0, 2)  # move axis to end


def im_to_int(im):
    """
    Transforms image into int8. Zeroes are ignored.
    :param im: A 3-channel image.
    :return:  A 3-channel np.uint8 image.
    """
    max = np.max(im)
    min = np.min(im[im > 0])
    out = (255 * ((im - min) / (max - min))).astype(np.float32)
    out[out < 0] = 0
    return out.astype(np.uint8)


def make_mosaic(im_list):
    shape = int(np.ceil(np.sqrt(len(im_list))))
    resize = int(1400 / shape)
    resized_list = [imutils.resize(im_to_int(im), width=resize, height=resize)[:resize - 1, :resize - 1] for im in
                    im_list]
    outarr = np.zeros((shape * (resize - 1), shape * (resize - 1)), dtype=np.uint8)
    for i, im in enumerate(resized_list):
        ind = np.unravel_index(i, (shape, shape))
        outarr[ind[0] * (resize - 1):(ind[0] + 1) * (resize - 1),
        ind[1] * (resize - 1):(ind[1] + 1) * (resize - 1)] = im
    return outarr


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
