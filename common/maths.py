import numpy as np
import cv2


def safe_divide(x, y):
    """
    Divides x by y, catching ZeroDivisionErrors and returning None if dividing by zero.
    :param x: Division numerator.
    :param y: Division denominator.
    :return: A float or None.
    """
    try:
        return float(x)/float(y)
    except ZeroDivisionError:
        print("Warning: Dividing by zero. Returning None.")
        return None


def safe_round(x, d=3):
    if x is not None:
        return np.round(x, d)
    else:
        return None


def normalize(a, nmin=0, nmax=1):
    out = (a-np.min(a))/(np.max(a)-np.min(a))
    out = nmin + (out * (nmax-nmin))
    return out


def normalized_diff(a, b):
    """
    Normalized difference function. i.e. NDVI.
    :param a: A float or 1D or 2D float array.
    :param b: A float or 1D or 2D float array. Same dims as a.
    :return: A float or array of the same dims as a/b.
    """
    return (a-b)/(a+b)


def get_float_channels(rgbim, mode='rgb'):
    a = np.array(rgbim).astype(np.float32)
    if mode == 'rgb':
        r = a[:, :, 0]
        g = a[:, :, 1]
        b = a[:, :, 2]
    elif mode == 'bgr':
        r = a[:, :, 2]
        g = a[:, :, 1]
        b = a[:, :, 1]
    return r, g, b


def vari(rgbim):
    """
    Calculate the Visible Atmospherically Resistant Index (VARI) for an RGB image.

    VARI is a vegetation index that can be used to estimate vegetation health and density.
    It is calculated using the formula: (G - R) / (G + R - B), where R, G, and B are the red,
    green, and blue channels of the input RGB image.

    Args:
    - rgbim (numpy.ndarray): A 3-dimensional numpy array representing an RGB image.

    Returns:
    - numpy.ndarray: A 2-dimensional numpy array representing the VARI values calculated for
    the input RGB image. Values outside of the range [-1, 1] are clipped to -1 and 1, respectively.
    """
    r, g, b = get_float_channels(rgbim)
    vari = (g-r)/(g+r-b)
    vari[vari < -1] = -1
    vari[vari > 1] = 1
    return vari


def red_ratio(rgbim):
    r, g, b = get_float_channels(rgbim)
    return 3*r/(r+g+b)


def blue_ratio(rgbim):
    r, g, b = get_float_channels(rgbim)
    return 3*b/(r+g+b)


def green_ratio(rgbim):
    r, g, b = get_float_channels(rgbim)
    return 3*g/(r+g+b)


def var_img(rgbim):
    a = np.array(rgbim).astype(np.float)
    return np.var(a, axis=-1)


def gray(rgbim):
    return cv2.cvtColor(rgbim, cv2.COLOR_RGB2GRAY)
