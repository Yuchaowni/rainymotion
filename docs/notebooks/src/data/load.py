"""
    File name: load.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-04-01
    Python Version: 3.6
"""


import numpy as np
import cv2


def binary(filename, shape, dtype='float32'):
    """
    Read binary data (particularly for Ikai-san's precipitation data)
    :param filename: str, file name
    :param shape: tuple (row, col), the shape of 2d data
    :param dtype: str, data type
    :return: np.ndarray, data
    """
    data = np.fromfile(filename, dtype=dtype)
    data = data.astype(np.float64)
    data = data.reshape(*shape)
    data = cv2.blur(data, (10, 10))
    return data

