from common.rgb_image import RGBImage
from common.maths import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

LOC = r'C:\Users\Kristof\Desktop\SESVanderhave\Virus Yellows\image data\IMG_3692.jpg'


class YellowImage(RGBImage):
    def yellow_index(self):
        """
        Calculates the yellow index for the underlying RGB image.
        :return: 2d grayscale array.
        """
        ch = self.get_channels()

        yellow_distance = np.sqrt(ch['B']**2+(256-ch['R'])**2+(256-ch['G'])**2)
        yellow_intensity = (450-yellow_distance)/450
        return yellow_intensity

    def dominant_channel(self):
        """
        Calculates the channel with the largest value for each pixel.
        :return: 2d array with integer values ranging from 0 to 2.
        """
        return np.argmax(self.float_im, axis=-1)

    def color_anisotropy(self):
        """
        Calculates color anisotropy. Essentially a value of how one channel is overrepresented vs. others.
        :return: 2d grayscale array with float values between 0 and 1.
        """
        ch = self.get_channels()
        r = ch['R']
        g = ch['G']
        b = ch['B']

        ani = (np.sqrt(1/2)) * (np.sqrt((r-g)**2 + (g-b)**2 + (b-r)**2)/np.sqrt(r**2 + g**2 + b**2))
        return ani

    def vari(self):
        """Calculates the Visible Atmospherically Resistant Index (VARI) from a 2D image
        and returns a VARI map.
        :return: 2d grayscale array with float values between -1 and 1."""
        ch = self.get_channels()
        r = ch['R']
        g = ch['G']
        b = ch['B']

        vari = (g-r)/(g+r-b)
        vari[vari<-1] = -1
        vari[vari>1] = 1
        return vari

    def tgi(self):
        """Calculates the Triangular Greenness Index from a 2D image
        and returns a TGI map.
        :return: 2d grayscale array"""
        ch = self.get_channels()
        r = ch['R']/256
        g = ch['G']/256
        b = ch['B']/256

        # central wavelengths per channel in nm
        lr = 670
        lg = 550
        lb = 480

        tgi = -0.5 * (((lr-lb)*(r-g)) - ((lr-lg)*(r-b)))
        return tgi

    def green_ratio(self):

        ch = self.get_channels()

        green_intensity = (ch['G']) / self.hsv[:, :, 2]
        return green_intensity

    def green_mask(self, hsv_mask = [(36, 50, 25), (70, 255,255)]):
        mask = cv2.inRange(self.hsv, hsv_mask[0], hsv_mask[1])
        return mask

    def normalized_green(self):
        ch = self.get_channels()

        ndgi = normalized_diff(ch['G'], (ch['R']+ch['B'])/2)
        return ndgi

    def show_yellow_intensity(self, vmin=0, vmax=1):
        plt.figure()
        plt.imshow(self.yellow_index(), vmin=vmin, vmax=vmax)
        plt.title('Yellow Intensity')
        plt.colorbar()

    def show_yellow_and_green(self, yellow_thresh=0.66):
        gm = self.green_mask()
        a = np.zeros(gm.shape)
        ym = np.where(self.yellow_index() > yellow_thresh, 255, 0)
        a[gm > 0] = 1
        a[ym > 0] = 2
        a[a == 0] = np.nan

        plt.figure()
        plt.subplot(1, 2, 1)
        self.show()
        plt.subplot(1, 2, 2)
        plt.imshow(self.gray, cmap='gray')
        plt.imshow(a, alpha=0.6, cmap='rainbow')


def summarize_channels(rgb_im, thresh=0):
    r, g, b = [rgb_im[:, :, i] for i in range(3)]
    r1 = np.mean(r[r > thresh])
    r2 = np.std(r[r > thresh])
    g1 = np.mean(g[g > thresh])
    g2 = np.std(g[g > thresh])
    b1 = np.mean(b[b > thresh])
    b2 = np.std(b[b > thresh])
    return r1, r2, g1, g2, b1, b2


if __name__ == '__main__':
    LOC = r'C:\Users\Kristof\Desktop\SESVanderhave\Virus Yellows\image data\panorama\IMG_3445.JPG'
    yi = YellowImage(LOC)
    yi.show_yellow_and_green()
