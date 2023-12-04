import cv2
import numpy as np
import matplotlib.pyplot as plt


class RGBImage:
    def __init__(self, location: str):
        self.im = cv2.cvtColor(cv2.imread(location), cv2.COLOR_BGR2RGB)
        self.float_im = 1.0 + np.ndarray.astype(self.im, 'float')
        self.hsv = np.ndarray.astype(cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV), 'float')
        self.gray = np.ndarray.astype(cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY), 'float')
        self.shape = self.im.shape
        self.range = (np.min(self.im), np.max(self.im))
        self.order = 'BGR'

    def get_channels(self):
        r = self.float_im[:, :, self.order.find('R')]
        g = self.float_im[:, :, self.order.find('G')]
        b = self.float_im[:, :, self.order.find('B')]

        return {
            "R": r,
            "G": g,
            "B": b
        }

    def get_rgb_order(self):
        return (self.order.find('R'),
                self.order.find('G'),
                self.order.find('B'))

    def show(self):
        plt.imshow(self.im[:, :, self.get_rgb_order()])

    def show_ch(self, ch):
        plt.figure()
        plt.imshow(self.im[:,:,ch])
        plt.title('Channel: ' + self.order[ch])
        plt.colorbar()

    @classmethod
    def pixel_fraction(cls, im2d, thresh):
        return np.mean(np.where(im2d > thresh, 1.0, 0.0))

    def show_with_overlay(self, im2d):
        plt.figure()
        plt.subplot(1, 2, 1)
        self.show()
        plt.subplot(1, 2, 2)
        plt.imshow(self.gray, cmap='gray')
        plt.imshow(im2d, alpha=0.6, cmap='jet')
