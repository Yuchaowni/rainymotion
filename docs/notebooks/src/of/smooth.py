"""
    File name: smooth.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-03-27
    Python Version: 3.6
"""


import numpy as np
import cv2


def _svd(img, k):
    imgu, imgs, imgv = np.linalg.svd(img)
    imguk, imgvk = imgu[:, :k], imgv[:k, :]
    imgsk = np.zeros((k, k))
    for i in range(k):
        imgsk[i, i] = imgs[i]
    imgk = np.dot(np.dot(imguk, imgsk), imgvk)
    return imgk


def _cumsum(img, axis):
    img_res = np.zeros_like(img, dtype=np.float64)
    if axis == 0:
        for i in range(img.shape[0]):
            img_res[i] = np.sum(img[:i], axis=0)
    elif axis == 1:
        for i in range(img.shape[1]):
            img_res[:, i] = np.sum(img[:, :i], axis=1)
    return img_res


def _repeat(img, n, axis):
    if axis == 0:
        img_res = np.zeros((n, img.shape[0]))
        for i in range(n):
            img_res[i] = img
    elif axis == 1:
        img_res = np.zeros((img.shape[0], n))
        for i in range(n):
            img_res[:, i] = img
    return img_res


def _ma(img, window):
    r = int(min(*window) // 2)
    hei = img.shape[0]
    wid = img.shape[1]
    img_res = np.zeros((hei, wid))

    # along axis=0
    img_cum = _cumsum(img, 0)
    img_res[:r+1, :] = img_cum[r:2*r+1, :]
    img_res[r+1:hei-r, :] = img_cum[2*r+1:hei, :] - img_cum[:hei-2*r-1, :]
    img_res[hei-r:, :] = _repeat(img_cum[hei-1, :], r, 0) - img_cum[hei-2*r-1:hei-r-1, :]

    # along axis=1
    img_cum = _cumsum(img_res, 1)
    img_res[:, :r+1] = img_cum[:, r:2*r+1]
    img_res[:, r+1:wid-r] = img_cum[:, 2*r+1:wid] - img_cum[:, :wid-2*r-1]
    img_res[:, wid-r:] = _repeat(img_cum[:, wid-1], r, 1) - img_cum[:, wid-2*r-1:wid-r-1]
    return img_res


def moving_average(img, window, max_iter):
    for i in range(max_iter):
        window = tuple(window)
        img = cv2.blur(img, window)
    return img


def fill_average(img, th=1e-2):
    img_abs = np.abs(img)
    cond = img_abs < th
    img[cond] = img[~cond].mean() * 1.5
    # img[cond] = np.median(img[~cond])
    return img


def fill_moving_average(img, window, max_iter, th=1e-2):
    window = tuple(window)
    img_abs = np.abs(img)
    for i in range(max_iter):
        if th is None or th == 0:
            avg = img_abs.mean()
            std = img_abs.std()
            th = avg - std * 2
        img_ma = cv2.blur(img, window)
        img_ma[img_abs > th] = 0.
        img += img_ma
    return img


def svd(img, k=2, keep_ratio=.5):
    img_svd = _svd(img, k)
    img_res = img_svd * keep_ratio + img_svd * (1 - keep_ratio)
    return img_res


def fill_svd(img, k=2, th=1e-3):
    img_svd = _svd(img, k)
    img_abs = np.abs(img)
    cond = img_abs < th
    img[cond] = np.mean(img_svd[~cond]) * 1.5
    # img[cond] = np.median(img_svd[~cond])
    return img


