"""Plotting helpers for the app. Pure functions: data in, figure or chart out."""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tic_tac_fly.brain.connectome import Connectome
from tic_tac_fly.brain.reservoir import Reservoir
from tic_tac_fly.game.board import Game

# Ellipsoid body cell families, in display order
FAMILIES = ("ER", "ExR", "EPG", "EPGt", "PEN", "PEG", "EL")


def family_of(cell_type: str) -> str:
    """Cell family of a FlyWire cell type, e.g. "ER4d" -> "ER"."""
    # Longest prefix first, so "EPGt1" is EPGt rather than EPG
    return next(
        (f for f in sorted(FAMILIES, key=len, reverse=True) if cell_type.startswith(f)),
        "other",
    )


def neuron_groups(connectome: Connectome) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """A neuron order grouped by cell family, and the (family, start, stop) bands."""
    if connectome.cell_types is None:
        return np.arange(connectome.n_neurons), [("all neurons", 0, connectome.n_neurons)]
    families = np.array([family_of(str(t)) for t in connectome.cell_types])
    rank = {family: i for i, family in enumerate((*FAMILIES, "other"))}
    # Stable sort keeps FlyWire's order within each family
    order = np.argsort([rank[f] for f in families], kind="stable")
    grouped = families[order]
    bands, start = [], 0
    for family in dict.fromkeys(grouped):  # unique, in order
        stop = start + int((grouped == family).sum())
        bands.append((family, start, stop))
        start = stop
    return order, bands


def label_bands(axis, bands: list[tuple[str, int, int]], which: str) -> None:
    """Tick labels at the middle of each family band, lines between bands."""
    centres = [(start + stop - 1) / 2 for _, start, stop in bands]
    names = [name for name, _, _ in bands]
    line = axis.axhline if which == "y" else axis.axvline
    for _, start, _ in bands[1:]:
        line(start - 0.5, color="black", lw=0.5)
    if which == "y":
        axis.set_yticks(centres, names, fontsize=7)
    else:
        axis.set_xticks(centres, names, fontsize=7, rotation=90)


def value_grid(values: np.ndarray, outlined: frozenset[int], title: str) -> alt.Chart:
    """3x3 heatmap of move values (red good for the fly), outlining some squares."""
    df = pd.DataFrame({
        "row": np.arange(9) // 3,
        "col": np.arange(9) % 3,
        "value": values,
        "label": ["" if np.isnan(v) else f"{v:+.2f}" for v in values],
        "outlined": [square in outlined for square in range(9)],
    })
    base = alt.Chart(df, title=title).encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None)
    )
    cells = base.mark_rect(stroke="black").encode(
        color=alt.Color(
            "value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=None
        ),
        strokeWidth=alt.condition("datum.outlined", alt.value(4), alt.value(0)),
    )
    text = base.mark_text(fontSize=16).encode(text="label:N")
    return (cells + text).properties(width=170, height=170)


def raster_figure(steps: np.ndarray, order: np.ndarray, bands, game: Game, steps_per_move: int):
    """Neurons (grouped by family) x time steps, with a line at each move."""
    figure, axis = plt.subplots(figsize=(8, 3.4))
    axis.imshow(
        steps[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
    )
    for k, move in enumerate(game.moves):
        # Step 0 is rest; move k occupies steps k*s+1 ... (k+1)*s
        boundary = k * steps_per_move + 0.5
        axis.axvline(boundary, color="black", lw=0.8)
        axis.text(
            boundary + steps_per_move / 2, 1.02, f"{move.player.symbol}{move.square}",
            transform=axis.get_xaxis_transform(), ha="center", fontsize=8,
        )
    label_bands(axis, bands, "y")
    axis.set_xlabel("time step")
    figure.tight_layout()
    return figure


def connectivity_figure(connectome: Connectome, order: np.ndarray, bands):
    """Signed synapse counts, log-scaled, grouped by family.

    Rows receive, columns send. Red columns are excitatory senders, blue
    inhibitory.
    """
    weights = connectome.weights[np.ix_(order, order)]
    # Signed log scale, so both a few and hundreds of synapses are visible
    scaled = np.sign(weights) * np.log1p(np.abs(weights))
    limit = np.abs(scaled).max() or 1.0
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(scaled, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    label_bands(axis, bands, "y")
    label_bands(axis, bands, "x")
    axis.set_xlabel("sending neuron")
    axis.set_ylabel("receiving neuron")
    figure.tight_layout()
    return figure


def eigenvalue_chart(reservoir: Reservoir) -> alt.Chart:
    """Eigenvalues of W in the complex plane, with reference circles."""
    eigenvalues = reservoir.eigenvalues
    radius = reservoir.config.spectral_radius
    norm = reservoir.operator_norm
    angle = np.linspace(0, 2 * np.pi, 200)
    circles = pd.concat([
        pd.DataFrame({
            "real": r * np.cos(angle), "imaginary": r * np.sin(angle),
            "circle": name, "order": np.arange(len(angle)),
        })
        for name, r in (
            ("unit circle", 1.0),
            (f"spectral radius {radius:.2f}", radius),
            (f"operator norm {norm:.2f}", norm),
        )
    ])
    # Same range on both axes, so circles look round
    scale = alt.Scale(domain=[-max(norm, 1.0) * 1.05, max(norm, 1.0) * 1.05])
    rings = alt.Chart(circles).mark_line(strokeDash=[4, 3]).encode(
        x=alt.X("real:Q", scale=scale), y=alt.Y("imaginary:Q", scale=scale),
        color=alt.Color("circle:N", legend=alt.Legend(orient="bottom", title=None)),
        order="order:Q",
    )
    points = alt.Chart(
        pd.DataFrame({"real": eigenvalues.real, "imaginary": eigenvalues.imag})
    ).mark_circle(size=14, opacity=0.6, color="black").encode(
        x=alt.X("real:Q", scale=scale), y=alt.Y("imaginary:Q", scale=scale)
    )
    return (rings + points).properties(width=330, height=330)