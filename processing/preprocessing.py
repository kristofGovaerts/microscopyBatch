"""
code goes here
"""

import numpy as np
import cv2
import vidstab
import lensfunpy
from scipy.ndimage import morphology
from scipy.signal import convolve2d


def normalize(arr):
    """
    Rescales an array from 0 to 1.
    :param vec: np.ndarray
    :return: Normalized np.ndarray
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def sharpen(arr2d, kernel=np.array([[-1,-1,-1],[-1,12,-1],[-1,-1,-1]])):
    return convolve2d(arr2d, kernel)[1:-1, 1:-1]


def align_images(image, template,
                 maxFeatures=500,
                 keepPercent=0.2,
                 debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


def stabilize_images(im_list):
    stabilizer = vidstab.VidStab()
    for i in range(len(im_list)):
        im = cv2.imread(im_list[i])
        b = red_ratio(im)
        im[b<1.5] = 0
        s = stabilizer.stabilize_frame(input_frame=im, smoothing_window=1)
        if i == 0:
            cv2.imwrite(str(i) + '.JPG', im)
        else:
            cv2.imwrite(str(i) + '.JPG', s)


def remove_small_objects(img, min_size=100):
    """
    Uses connected component analysis to remove objects below a certain size threshold.
    :param img: A binary mask.
    :param min_size: Minimum allowed component size.
    :return: A filtered mask.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = img.copy()
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0
    return img2


def remove_objects_by_size(img, min_size=1, max_size=np.inf):
    """
    Uses connected component analysis to remove objects below a certain size threshold.
    :param img: A binary mask.
    :param min_size: Minimum allowed component size.
    :param max_size: Maximum allowed component size.
    :return: A filtered mask.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size or sizes[i] > max_size:
            img2[output == i + 1] = 0
    return img2



def remove_spatial_trend(image, filter_size, highpass=False, norm=True):
    smoothed = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    detrended = image - smoothed

    if highpass:
        gradient_x = cv2.Sobel(detrended, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(detrended, cv2.CV_64F, 0, 1, ksize=3)
        detrended = cv2.add(gradient_x, gradient_y)

    if norm:
        detrended = normalize(detrended)

    return detrended


def correct_lens_distortion(im, cam_maker='GOPRO', cam_model='Hero7 Black',
                            focal_length=3, aperture=2.97, distance=0, db=None):
    height, width, c = im.shape

    if db is None:
        db = lensfunpy.Database()
    if cam_maker == 'GOPRO':
        try:
            cam = db.find_cameras(maker='GOPRO', model=cam_model)[-1]  # most recent GOPRO camera
        except IndexError:
            try:
                cam = db.find_cameras(maker='GOPRO')[-1]  # most recent GOPRO camera
            except IndexError:
                print("Camera not found!")
        lens = db.find_lenses(cam)[0]
    print("""
    Camera: {}
    Lens:   {}""".format(cam, lens))
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)
    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted
