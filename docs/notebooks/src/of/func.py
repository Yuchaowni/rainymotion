"""
    File name: func.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2018-12-19
    Python Version: 3.6
"""


import numpy as np
import cv2


def im_resize(dsize, direction, *args):
    res = []
    for img in args:
        if direction == 'down':
            img_d = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        elif direction == 'up':
            img_d = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        else:
            raise Exception('Unrecognized sampling direction')
        res.append(img_d)
    return res


def im_affine(threshold, *args):
    res = []
    for img in args:
        img_tmp = img.copy()
        img_tmp[img_tmp < threshold] = 0.
        res.append(img_tmp)
    return res


def calc_differential(img, imgp, dx, dy, dt, robust_mode, **kwargs):
    """ simple difference method, NEED TO UPDATE ???
    :param img: np.ndarray, 初期(レーダ)画像
    :param imgp: np.ndarray, 初期画像の一つ前の画像
    :param dx: np.ndarray, 偏微分におけるdxの値
    :param dy: np.ndarray, 偏微分におけるdyの値
    :param dt: np.ndarray, 偏微分におけるdtの値
    :param robust_mode: int, mode of robust estimation; 0 if not using robust estimation
    :return:
    """
    wx = np.zeros_like(img, dtype=np.float64)
    wy = np.zeros_like(img, dtype=np.float64)
    wx[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, :-2]
    wy[1:-1, 1:-1] = img[2:, 1:-1] - img[:-2, 1:-1]
    rx = wx / (dx * 2)
    ry = wy / (dy * 2)

    wt = img - imgp
    rt = wt / dt

    if robust_mode != 0:
        rx, ry, rt = robust(robust_mode, rx, ry, rt,
                            wx, wy, wt,
                            **kwargs)

    return rx, ry, rt


def boundary(img):
    """2次元配列のテーブルで最外周の領域に内周の値を与える
    :param img: input image
    :return:
    """
    # left and right
    img[:, 0] = img[:, 1]
    img[:, -1] = img[:, -2]
    # top and bottom
    img[0, :] = img[1, :]
    img[-1, :] = img[-2, :]
    # corners
    img[0, 0] = img[1, 1]
    img[0, -1] = img[1, -2]
    img[-1, 0] = img[-2, 1]
    img[-1, -1] = img[-2, -2]
    return img


def robust(mode, rx, ry, rt, wx, wy, wt, **kwargs):
    """
    :param mode: int, 1, 2, or 3
    :param rx: np.ndarray, 微分値
    :param ry: np.ndarray, 微分値
    :param rt: np.ndarray, 微分値
    :param wx: np.ndarray, 画像X方向差分値(現在 img(x+1)-img(x-1) )
    :param wy: np.ndarray, 画像Y方向差分値(現在 img(x+1)-img(x-1) )
    :param wt: np.ndarray, 画像時間差分値(現在−過去)
    :param kwargs: dict, optional parameters for implementing each mode
    :return:
    """
    if mode == 1:
        # condition 1
        cond = wt[1:-1, 1:-1] <= 20. * 9.
        wt[1:-1, 1:-1][cond] = (1. - wt[1:-1, 1:-1][cond] / 20. / 9. ** 2) ** 2
        wt[1:-1, 1:-1][~cond] = 0.

        # condition 2
        cond1 = wt[1:-1, 1:-1] > 1.
        cond2 = wt[1:-1, 1:-1] < -1.
        wt[1:-1, 1:-1][cond1] = 1.
        wt[1:-1, 1:-1][cond2] = -1.

        # final
        rx[1:-1, 1:-1] *= wt[1:-1, 1:-1]
        ry[1:-1, 1:-1] *= wt[1:-1, 1:-1]
        rt[1:-1, 1:-1] *= wt[1:-1, 1:-1]

    elif mode == 2:
        threshold = kwargs.get('robust_threshold', 10.)
        cond = wt[1:-1, 1:-1] > threshold
        rx[1:-1, 1:-1][cond] = 0.
        ry[1:-1, 1:-1][cond] = 0.
        rt[1:-1, 1:-1][cond] = 0.

    elif mode == 3:
        # get optional parameters for mode 3
        threshold = kwargs.get('robust_threshold', 10.)
        xy = kwargs.get('process_xy', True)
        x_coef = kwargs.get('cx', 10.)
        y_coef = kwargs.get('cy', 10.)
        t_coef = kwargs.get('ct', 2.)

        # rx and ry
        if xy is True:
            # x
            wx_tmp = wx[:, 1:-1]
            wx_tmp = np.abs(wx_tmp[wx_tmp != 0.])
            x_med = np.median(wx_tmp)
            cond = wx[1:-1, 1:-1] > x_med * x_coef
            rx[1:-1, 1:-1][cond] *= np.cos((np.pi / 2) * (np.abs(wx[1:-1, 1:-1][cond]) / 255.))
            # y
            wy_tmp = wy[1:-1, :]
            wy_tmp = np.abs(wy_tmp[wy_tmp != 0.])
            y_med = np.median(wy_tmp)
            cond = wy[1:-1, 1:-1] > y_med * y_coef
            ry[1:-1, 1:-1][cond] *= np.cos((np.pi / 2) * (np.abs(wy[1:-1, 1:-1][cond]) / 255.))

        # rt
        wt_tmp = np.abs(wt[wt != 0.])
        t_med = np.median(wt_tmp)
        cond1 = wt[1:-1, 1:-1] >= threshold
        cond2 = np.abs(wt[1:-1, 1:-1]) > t_med * t_coef
        cond2 = (~cond1) & cond2
        rt[1:-1, 1:-1][cond1] = 0.
        rt[1:-1, 1:-1][cond2] *= np.cos((np.pi / 2) * (np.abs(wt[1:-1, 1:-1][cond2]) / 255.))

    return rx, ry, rt


def vel_affine(uh, vh, xoff, yoff):
    """ ???
    :param uh:
    :param vh:
    :param xoff:
    :param yoff:
    :return:
    """
    dm = 6
    dy = 2 * yoff + 1
    dx = 2 * xoff + 1

    an = np.zeros(dm, dtype=np.float64)
    w = np.zeros(dm, dtype=np.float64)
    bg = np.zeros(dm, dtype=np.float64)

    cx = np.ones((dy, dx))

    cy = np.ones((dy, dx))
    cx = np.cumsum(cx, axis=1)
    cy = np.cumsum(cy, axis=0)

    xoff2param_iter = {
        10: 1e-3,
        20: 1e-4,
        30: 1e-5,
        40: 1e-6,
        50: 1e-7,
        60: 1e-8
    }  # 10x10:0.001	30x30:0.0001  40x40:0.00001
    param_iter = xoff2param_iter[xoff * 2]

    for i in range(100):
        eng = 0
        w = np.zeros(dm, dtype=np.float64)

        if i < 10:
            ua = an[0]
            va = an[3]
            an[1, 2, 4, 5] = 0.
        else:
            ua = an[0] + an[1] * cx + an[2] * cx
            va = an[3] + an[4] * cy + an[5] * cy
        bufx = uh - ua
        bufy = vh - va

        w[0] = np.sum(bufx)
        w[1] = np.sum(bufx * cx)
        w[2] = np.sum(bufx * cy)
        w[3] = np.sum(bufy)
        w[4] = np.sum(bufy * cx)
        w[5] = np.sum(bufy * cy)

        eng = np.sum(bufx ** 2 + bufy ** 2)

        cond = w != 0
        an[cond] += param_iter * w[cond] / (dy * dx)  # ??? QUESTION, NEED TO CONFIRM

    u_affine = np.sum(an[:3] * [1, dx/2., dy/2.])
    v_affine = np.sum(an[3:] * [1, dx/2., dy/2.])

    return u_affine, v_affine

