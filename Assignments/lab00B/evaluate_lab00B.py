# -*- coding: utf-8 -*-
"""Evaluate Lab 00B — Gaussian Kernels, Convolution & Interpolation.

Run from the repo root:
    python Assignments/lab00B/evaluate_lab00B.py

Results are written to a single HTML file next to this script.
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LAB_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab00B import ConvolutionInterpolator
from helpers.plotting import save_html


# ── Demo parameters ──────────────────────────────────────────────────────

PATTERN_SIZE = 500
F0 = 10
CHIRP_RATE = 250

SIGMA = 2.5
SCALE_FACTOR = 0.2
ORDER = 0


def _make_chirp_pattern(
    size=PATTERN_SIZE,
    f0=F0,
    chirp_rate=CHIRP_RATE
):
    """Generate a 2D chirp ("zone plate") test pattern.

    a(x, y) = sin(2*pi*(f0*R + chirp_rate*R**2 / 2))

    where R = sqrt(x**2 + y**2).
    """

    coords = np.arange(size) / size - 0.5

    X, Y = np.meshgrid(coords, coords)

    R = np.sqrt(X ** 2 + Y ** 2)

    return np.sin(
        np.pi * 2 * (
            f0 * R +
            chirp_rate * R ** 2 / 2
        )
    )


def plot_aliasing_comparison(original, aliased, filtered):
    """Build a 3-panel Plotly figure comparing:

    1. Original pattern
    2. Direct interpolation
    3. Gaussian-filtered interpolation
    """

    titles = [
        "Original pattern",
        "Direct interpolation (aliased)",
        "Gaussian-filtered interpolation"
    ]

    panels = [
        original,
        aliased,
        filtered
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=titles
    )

    for col, panel in enumerate(panels, start=1):

        fig.add_trace(
            go.Heatmap(
                z=panel,
                colorscale="gray",
                showscale=False
            ),
            row=1,
            col=col,
        )

    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_layout(
        title_text="Lab 00B: Aliasing vs. Anti-Aliased Interpolation",
        height=350
    )

    return fig


def run_aliasing_demo(pattern):
    """Run both interpolation paths.

    First:
        Direct interpolation without filtering.

    Second:
        Gaussian filtering followed by interpolation.

    Args:
        pattern (ndarray):
            H x W test pattern.

    Returns:
        aliased (ndarray):
            Directly interpolated pattern.

        filtered (ndarray):
            Gaussian-prefiltered and then interpolated pattern.
    """

    # Create the interpolator.
    interpolator = ConvolutionInterpolator()

    # ---------------------------------------------------------
    # 1. Direct interpolation
    # ---------------------------------------------------------

    aliased = interpolator.interpolate(
        pattern,
        scale_factor=SCALE_FACTOR,
        order=ORDER
    )

    # ---------------------------------------------------------
    # 2. Gaussian pre-filtering
    # ---------------------------------------------------------

    kernel = interpolator.gaussian_kernel_2d(
        SIGMA
    )

    smoothed = interpolator.convolve(
        pattern,
        kernel
    )

    # ---------------------------------------------------------
    # 3. Interpolate the filtered image
    # ---------------------------------------------------------

    filtered = interpolator.interpolate(
        smoothed,
        scale_factor=SCALE_FACTOR,
        order=ORDER
    )

    return aliased, filtered


def evaluate():

    print("Generating chirp test pattern...")

    pattern = _make_chirp_pattern()

    print(
        "Running interpolation with and without a Gaussian pre-filter..."
    )

    aliased, filtered = run_aliasing_demo(pattern)

    print("Building comparison figure...")

    fig = plot_aliasing_comparison(
        pattern,
        aliased,
        filtered
    )

    save_html(
        fig,
        output_path=os.path.join(
            LAB_DIR,
            "lab00B_results.html"
        )
    )

    print(
        "Results saved →",
        os.path.join(
            LAB_DIR,
            "lab00B_results.html"
        )
    )


if __name__ == "__main__":
    evaluate()