import os
import cv2
import numpy as np
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import astropy.units as u
from fil_finder import FilFinder2D
from processing.segmentation import find_blob_centers
from processing.preprocessing import remove_objects_by_size, remove_spatial_trend, normalize
from common.io import get_images_in_folder

EXT = '*.tif'
THRESH_FLUO = 0.15
THRESH_RGB = 0.05
CELL_DIAM = 10  # approximate diameter of a cell, in pixels
LABEL_COLOR = (0, 255, 0)  # color for annotations in RGB format
TAIL_COLOR = (255, 0, 0)  # color for annotations in RGB format
DETREND = True

# size threshold based on average cell surface- filter out noise and double cells
cell_size = (CELL_DIAM/2)**2 * 3.14
SIZE_THRESH = [cell_size/5, 1.9*cell_size]

if __name__ == '__main__':
    folder, files = get_images_in_folder(pattern=EXT)
    os.chdir(folder)

    out = []

    if not os.path.isdir('output'):
        os.mkdir('output')

    for i, f in enumerate(files):
        print("Image {} of {}:\n\t{}".format(i+1, len(files), f))
        im = cv2.imread(f)
        im = imutils.resize(im, width=1292)
        im = cv2.GaussianBlur(im, (5,5), 5)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv /= [180, 255, 255]  # normalize

        if np.median(hsv[:,:,2]) > 0.5:  # brightfield image, otherwise it's a fluo image
            print("Brightfield image. Inverting.")
            MODE = 'brightfield'
            hsv[:, :, 2] = np.abs(hsv[:, :, 2]-1)  # invert
            THRESH = THRESH_RGB
            # intensity = hsv[:, :, 2] * hsv[:, :, 1]
            intensity = hsv[:, :, 2]
            intensity = cv2.GaussianBlur((255*normalize(intensity)).astype(np.uint8), (5,5), 5).astype(np.float32)/255
        else:
            if np.max(hsv[:, :, 2]) < 0.1:
                print("No fluorescence detected.")
                plt.figure()
                plt.imshow(im)
                plt.title(f + " / Count: {}".format(0))
                plt.savefig('output/' + f.replace(EXT[1:], '_annotated.tif'))
                plt.close()
                out.append([f, 0])
                continue

            MODE = 'fluorescence'
            THRESH = THRESH_FLUO
            print("Fluorescence image. No inversion.")
            intensity = hsv[:, :, 2]

        intensity[intensity<0.01] = 0
        if DETREND:
            intensity = remove_spatial_trend(intensity, filter_size=21, highpass=False, norm=False)
        intensity[intensity<0] = 0
        intensity = normalize(intensity)
        intensity[hsv[:, :, 1] < 0.1] = 0

        thresh_all = 0.1
        m_all = np.zeros((hsv.shape[0], hsv.shape[1]))
        m_all[intensity > thresh_all]=1

        # get cores & tails
        it=4
        cores = ndi.binary_dilation(ndi.binary_erosion(ndi.binary_fill_holes(m_all), iterations=it), iterations=it)
        cores = remove_objects_by_size(cores.astype(np.uint8), min_size=SIZE_THRESH[0], max_size=np.inf)
        tails = m_all.astype(np.uint8)-cores
        tails[tails!=1]=0
        tails = ndi.binary_erosion(ndi.binary_dilation(tails.astype(np.uint8), iterations=it), iterations=it).astype(np.uint8)
        tails = remove_objects_by_size(tails, min_size=5*SIZE_THRESH[0], max_size=np.inf)
        skel = skeletonize(ndi.binary_dilation(tails, iterations=3)).astype(np.uint8)

        # get live cells with a higher fluo threshold
        m = np.zeros_like(cores)
        m[(cores==1)&(intensity>THRESH_FLUO)] = 1
        m = ndi.binary_erosion(m.astype(np.uint8), iterations=3)
        m = remove_objects_by_size(m.astype(np.uint8), min_size=SIZE_THRESH[0], max_size=SIZE_THRESH[1])

        outim = 255*np.dstack([tails, m, cores-m])
        outim1, core_centers = find_blob_centers(m, to_label=outim, color=LABEL_COLOR)
        outim1, tail_centers = find_blob_centers(ndi.binary_dilation(skel, iterations=2).astype(np.uint8), to_label=outim1, color=TAIL_COLOR)

        print([len(core_centers), len(tail_centers)])
        plt.figure(figsize=(18,9))
        plt.imshow(np.hstack([im, outim1]))
        plt.title(f + " / Cells: {} / Tails: {}".format(len(core_centers), len(tail_centers)))
        if not os.path.isdir('output'):
            os.mkdir('output')
        plt.savefig('output/' + f.replace(EXT[1:], '_annotated.tif'))
        plt.close()
        out.append([f, len(core_centers), len(tail_centers)])
    df = pd.DataFrame(out, columns=['filename', 'cells', 'tails'])
    df.to_excel('output/' + 'blob_counts.xlsx', index=False)
