import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def imshow(img):
    print(img.shape)
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break


def pad(img, to_size=(128, 128, 3)):
    shape = img.shape
    if shape != to_size:
        out = np.zeros(to_size).astype(img.dtype)
        out[:shape[0],:shape[1],:shape[2]] = img
    else:
        out = img.copy()
    return out