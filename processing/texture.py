from processing.segmentation import crop
import cv2
import imutils
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

grayco_prop_list = ['contrast', 'dissimilarity',
                    'homogeneity', 'energy',
                    'correlation', 'ASM']


def get_tiles(im, factor=8):
    """
    Converts a 2D image to a grid of equally sized tiles. 'Factor' is the downscaling factor (i.e. downscaling an
    image with dimensions 128x128 by a factor of 8 produces a 16x16 grid of tiles.
    """
    xx, yy = np.meshgrid(
        np.arange(im.shape[1]),
        np.arange(im.shape[0]))
    region_labels = (xx // factor) * 64 + yy // factor
    region_labels = region_labels.astype(int)
    return region_labels


def to_textures(img, scale_factor=8, distances=[5], angles=[0], membership_threshold=0.66):
    """
    Takes an input 2D image and generates a texture map. Tile size is determined by scale_factor.

    :param img A 2D image with dtype np.uint8.
    :param scale_factor An int, determines tile size
    :param distances: Pixel pair distance offsets.
    :param angles: Pixel pair angles in radians.
    :param membership_threshold: fraction of overlap with mask necessary to count tile as foreground.
    :returns
        -A dictionary of property images with keys ['contrast', 'dissimilarity',
                                                    'homogeneity', 'energy',
                                                    'correlation', 'ASM']
        -A descriptive dataframe indicating stats per tile
        -A score image, indicating to what extent tiles correspond to the mask. 1 = fully within mask,
        0 = fully outside of mask,
    """
    im = np.copy(img)
    region_labels = get_tiles(im, factor=scale_factor)

    # generate binary mask, assuming 0 pixels are BG.
    slice_mask = np.zeros(im.shape, dtype=np.uint8)
    slice_mask[im != 0] = 1

    # Initialize output variables.
    prop_imgs = {}
    for c_prop in grayco_prop_list:
        prop_imgs[c_prop] = np.zeros_like(im, dtype=np.float32)

    out_df_list = []
    score_img = np.zeros_like(im, dtype=np.float32)

    # Iterate over image tiles
    for patch_idx in np.unique(region_labels):
        xx_box, yy_box = np.where(region_labels == patch_idx)

        glcm = graycomatrix(im[xx_box.min():xx_box.max(),
                            yy_box.min():yy_box.max()],
                            distances, angles, 256, symmetric=True, normed=True)

        mean_score = np.mean(slice_mask[region_labels == patch_idx])
        score_img[region_labels == patch_idx] = mean_score  # score indicates how much of a patch is part of the mask

        out_row = dict(
            intensity_mean=np.mean(im[region_labels == patch_idx]),
            intensity_std=np.std(im[region_labels == patch_idx]),
            score=mean_score)

        for c_prop in grayco_prop_list:
            out_row[c_prop] = graycoprops(glcm, c_prop)[0, 0]
            prop_imgs[c_prop][region_labels == patch_idx] = out_row[c_prop]

        out_df_list += [out_row]

        out_df = pd.DataFrame(out_df_list)
        out_df['positive_score'] = out_df['score'].map(lambda x: 'FG' if x > membership_threshold else 'BG')
    return prop_imgs, out_df, score_img


def texture_per_label(trait_im, label_im, resize_to=128, min_mask_size = 1000, scale_factor=8):
    out_dict = {}
    for label in np.unique(label_im):
        if label == 0:
            continue

        m = np.zeros(label_im.shape, dtype=np.uint8)
        m[label_im == label] = 1

        if np.sum(m) < min_mask_size:
            continue

        leaf = crop(trait_im * m)
        leaf[np.isnan(leaf)] = 0
        leaf = imutils.resize(leaf, width=resize_to, inter=cv2.INTER_NEAREST)

        # generate textures
        tim, tdf, sim = to_textures(leaf, scale_factor=scale_factor)

        # output table
        tdfs = tdf[tdf.columns[:-1]]
        normalized_df = (tdfs - tdfs.min()) / (tdfs.max() - tdfs.min())
        means = normalized_df[normalized_df['score'] > 0.66].mean()
        means_df = pd.DataFrame(np.reshape(np.array(means), (1, len(means))), columns=means.keys())
        means_df['label_idx'] = [label]
        if label == 1:
            out_df = means_df
        else:
            out_df = pd.concat([out_df, means_df])

        out_dict[str(label)] = [tim, sim, leaf]

    return out_dict, out_df