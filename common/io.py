import glob
import os
from tkinter import Tk, filedialog
import cv2


def get_images_in_folder(pattern='*.JPG', d=None):
    if d is None:
        root = Tk()
        fn = filedialog.askdirectory(initialdir="/",
                                     title="Select directory containing images")
        root.withdraw()
    else:
        fn = d
    ims = glob.glob1(fn, pattern)
    return fn, ims


def read_image(location, mode='rgb'):
    im = cv2.imread(location)
    if mode == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif mode == 'hsv':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    elif mode == 'lab':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    elif mode == 'bgr':
        im = im
    else:
        print("Image format not understood.")
    return im


def save_report(filename='output.csv', delimiter='\t', columns=['filename', 'total pixels',
                                                                'beetpix', 'beetfrac',
                                                                'greenpix', 'greenfrac',
                                                                'whitepix', 'whitefrac',
                                                                'darkpix', 'darkfrac',
                                                                'intensity']):
    with open(filename, 'w') as f:
        s = delimiter.join(columns) + '\n'
        f.write(s)


def add_line(pars, filename='output.csv', delimiter='\t'):
    with open(filename, 'a') as f:
        s = delimiter.join([str(p) for p in pars]) + '\n'
        f.write(s)
