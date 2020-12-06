import numpy as np
from PIL import Image

from numba import njit, prange


def binarize(img, window=25):
    img_arr = np.array(img, dtype=np.uint8)
    x, y, _ = img_arr.shape
    gray_img_arr = to_grayscale(img_arr)
    #threshold_arr = get_threshold_arr(gray_img_arr, window=window, method='niblack')
    #threshold_arr = get_threshold_arr(gray_img_arr, window=window, method='sauvola', k=0.2, r=100)
    threshold_arr = get_threshold_arr(gray_img_arr, window=window, method='cristian', k=0.1, r=100)
    #threshold_arr = get_threshold_arr(gray_img_arr, window=window, method='skew', k=0.01, r=256)
    #threshold_arr = get_threshold_arr(gray_img_arr, window=window, method='cristian_mod', k=0.2, r=128)
    binary_img = (gray_img_arr > threshold_arr).astype(np.int) * 255
    result_arr = np.zeros((x, y, 3), dtype=np.uint8)
    for i in range(3):
        result_arr[:, :, i] = binary_img
    return Image.fromarray(result_arr)


def to_grayscale(img_arr, method='luminosity'):
    gray_img_arr = np.zeros(img_arr.shape[:-1], dtype=np.uint8)
    if method == 'luminosity':
        gray_img_arr = 0.2125 * img_arr[:, :, 0] + 0.7154 * img_arr[:, :, 1] + 0.0721 * img_arr[:, :, 2]
    return gray_img_arr


@njit(parallel=True)
def get_threshold_arr(arr, window=50, method='niblack', k=0.0, r=0.0):
    radius = window // 2
    threshold_arr = np.zeros(arr.shape)
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[1]):
            min_x = max(i - radius - 1, 0)
            min_y = max(j - radius - 1, 0)
            max_x = min(i + radius + 1, arr.shape[0])
            max_y = min(j + radius + 1, arr.shape[1])
            segment = arr[min_x:max_x, min_y:max_y]
            mu = np.mean(segment)
            sigma = np.std(segment)
            if method == 'niblack':
                threshold_arr[i, j] = mu + k * sigma
            elif method == 'sauvola':
                threshold_arr[i, j] = mu * (1 + k * (sigma / r - 1))
            elif method == 'skew':
                skew = np.mean(np.power((segment - mu) / sigma, 3))
                threshold_arr[i, j] = mu * (1 + k * skew * (1 - skew / r))
            elif method == 'cristian':
                m = np.min(segment)
                threshold_arr[i, j] = (1 - k)*mu + k * m + k * sigma / r * (mu - m)
            elif method == 'cristian_mod':
                sigma_threshold = mu*0.07
                luminence_threshold = 20 + 0.66*mu
                delta = 5
                if sigma < sigma_threshold:
                    threshold_arr[i, j] = delta if mu > luminence_threshold else 255-delta
                else:
                    m = np.min(segment)
                    threshold_arr[i, j] = (1 - k)*mu + k * m + k * sigma / r * (mu - m)
    return threshold_arr
