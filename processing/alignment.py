import cv2
import numpy as np


def order_points(pts):
    """
    Helper function for ordering points from top left to bottom left, clockwise.
    :param pts: A 4x2 numpy array of points.
    :return: A 4x2 numpy array of points.
    """
    # order points: top-left clockwise to bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, new_dims):
    """
    Warps an image using a set of 4 points (pts), so that each point sits at a corner of the image.
    :param image: An image.
    :param pts: A set of 4 points located on the image. These points will become the 4 corner points of the warped img.
    :param new_dims: The desired output dimensions in (width, height).
    :return: The warped image.
    """
    rect = order_points(pts)
    maxWidth = new_dims[0]
    maxHeight = new_dims[1]

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped