"""
    File name: evaluate.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-02-17
    Python Version: 3.6
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle


def _of_vector_scale(u, v):
    u_med = np.nanmean(np.abs(u[u != 0]))
    v_med = np.nanmean(np.abs(v[v != 0]))
    scale = min(u_med, v_med) * 150
    return scale


def image(img, dt=None, layers=None,
          figsize=None, colorbar=False, ticks=False, gray=False, scalebar=False,
          patch_loc=None, patch_size=10):
    plt.clf()

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111)

    if layers is not None:
        if layers == 'default':
            layers = (0, .01, .05, .1, .5, 1, 3, 5, 10, 30)
        values = np.linspace(0, 255, len(layers)-1).astype(int)
        for i in range(len(layers)-1):
            img = img.copy()
            left = layers[i]
            right = layers[i+1]
            cond = (img >= left) & (img < right)
            img[cond] = values[i]

    if gray is True:
        img_out = ax.imshow(img, cmap='gray')
    else:
        img_out = ax.imshow(img)

    if colorbar is True:
        fig.colorbar(img_out)

    if ticks is False:
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    if dt is not None:
        if isinstance(dt, float) or isinstance(dt, int):
            hour = int(dt // 60)
            hour = ('0' + str(hour))[-2:]
            minutes = int(dt % 60)
            minutes = ('0' + str(minutes))[-2:]
            t = 'Time: %s:%s:00' % (hour, minutes)
            plt.text(1., 0., t, size=18)
        else:
            t = 'Time: %s' % dt
            plt.text(.7, .08, t, size=18)

    if patch_loc is not None:
        patch = Circle(patch_loc, radius=patch_size, color='red')
        ax.add_patch(patch)

    if scalebar is True:
        sb = ScaleBar(scalebar)  # built-in default unit is 1 pixel = 1 meter
        plt.gca().add_artist(sb)

    plt.tight_layout()

    return fig


def of_vector(u, v, img=None,  layers='default',
              scale=None, skip=10,
              figsize=None, dt=None, gray=False, scalebar=False, ticks=False,
              patch_loc=None, patch_size=10):
    plt.clf()

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111)

    ys, xs = u.shape
    u, v = u.copy(), v.copy()
    if skip is None:
        skip = int(min(u.shape) / 30)
    for y in range(ys):
        for x in range(xs):
            if y % skip == 0 and x % skip == 0 and x != 0 and y != 0:
                pass
            else:
                u[y, x] = np.nan
                v[y, x] = np.nan

    if img is not None:
        if layers is not None:
            if layers == 'default':
                layers = (0, .01, .05, .1, .5, 1, 3, 5, 10, 30)
            values = np.linspace(0, 255, len(layers) - 1).astype(int)
            for i in range(len(layers) - 1):
                img = img.copy()
                left = layers[i]
                right = layers[i + 1]
                cond = (img >= left) & (img < right)
                img[cond] = values[i]

        if gray is True:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        plt.gca().invert_yaxis()

    if scale is None:
        scale = _of_vector_scale(u, v)

    ax.quiver(u, -v, scale=scale, color='white')
    plt.gca().invert_yaxis()

    if ticks is False:
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    if dt is not None:
        if not isinstance(dt, str):
            hour = int(dt // 60)
            hour = ('0' + str(hour))[-2:]
            minutes = int(dt % 60)
            minutes = ('0' + str(minutes))[-2:]
            t = 'Time: %s:%s:00' % (hour, minutes)
            plt.text(1., 0., t, size=18)
        else:
            t = 'Time: %s' % dt
            plt.text(.7, .08, t, size=18)

    if patch_loc is not None:
        patch = Circle(patch_loc, radius=patch_size, color='red')
        ax.add_patch(patch)

    if scalebar is True:
        sb = ScaleBar(scalebar)  # built-in default unit is 1 pixel = 1 meter
        plt.gca().add_artist(sb)

    plt.tight_layout()

    return fig
