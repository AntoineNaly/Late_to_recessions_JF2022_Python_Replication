# -*- coding: utf-8 -*-

"""
recessionplot.py
----------------
Translated from MATLAB: recessionplot.m

Utility to overlay shaded NBER recession bands on a time series plot.

In the MATLAB code, the x-axis is in serial datenums. In Python we will
usually use datetime64 or integer index. We accept whatever the current
axes is using, and just shade vertical spans.
"""

import numpy as np
import matplotlib.pyplot as plt


def recessionplot(ax=None, recessions=None, facecolor="k", alpha=0.1):
    """
    Shade recession periods on an existing time-series plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to draw on. Defaults to current axis.
    recessions : array-like of shape (n_rec, 2) or None
        Each row is [start_x, end_x] in the SAME x-coordinate space
        as whatever is already plotted.
        If None, nothing will be drawn.
    facecolor : str or color
        Color of the recession band.
    alpha : float
        Transparency of the shading.

    Returns
    -------
    bands : list
        List of PolyCollections (the shaded spans).
    """
    if ax is None:
        ax = plt.gca()

    if recessions is None or len(recessions) == 0:
        return []

    bands = []
    xlim = ax.get_xlim()

    for start, end in np.asarray(recessions):
        # Only shade if interval overlaps the visible range
        if (start < xlim[1]) and (end > xlim[0]):
            band = ax.axvspan(
                start,
                end,
                ymin=0,
                ymax=1,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor="none",
            )
            bands.append(band)

    # Preserve original limits
    ax.set_xlim(*xlim)
    return bands


