"""The tabs: Play, Brain and Experiments."""

from __future__ import annotations

from typing import Callable

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tic_tac_fly.agent import FlyAgent, TrainingReport
from tic_tac_fly.brain.diagnostics import NetworkStats
from tic_tac_fly.brain.reservoir import ReservoirConfig
from tic_tac_fly.game.board import WINNING_LINES, Game, Player
from tic_tac_fly.game.minimax import best_squares
from tic_tac_fly.ui.cached import compare_variants, memory_curve, minimax_solver
from tic_tac_fly.ui.charts import (
    connectivity_figure,
    eigenvalue_chart,
    raster_figure,
    value_grid,
)

GLYPH = {0: " ", 1: "X", -1: "O"}


def render_play(
    game: Game,
    agent: FlyAgent,
    on_square: Callable[[int], None],
    on_reset: Callable[[], None],
) -> None:
    """Board, result, and the fly's last decision next to minimax."""
    board_column, decision_column = st.columns([1, 1.4])

    with board_column:
        st.caption("You: X, Fly: O. Fly's last move highlighted.")
        last = game.moves[-1].square if game.moves else None
        winning = next(
            (line for line in WINNING_LINES if abs(sum(game.board[i] for i in line)) == 3), ()
        )
        for row in range(3):
            cells = st.columns(3)
            for column in range(3):
                square = row * 3 + column
                cells[column].button(
                    GLYPH[game.board[square]], key=f"sq_{square}",
                    on_click=on_square, args=(square,), width="stretch",
                    type="primary" if square == last or square in winning else "secondary",
                )

        outcome = game.board.outcome()
        if outcome.winner is Player.X:
            st.success("You win")
        elif outcome.winner is Player.O:
            st.error("Fly wins")
        elif outcome.is_draw:
            st.info("Draw")

        st.button("New game", on_click=on_reset, width="stretch")
        if game.moves:
            st.caption("Moves: " + " ".join(f"{m.player.symbol}{m.square}" for m in game.moves))

    with decision_column:
        fly_turns = [k for k, move in enumerate(game.moves) if move.player is Player.O]
        # Rebuild the position the fly faced before its latest move
        k = fly_turns[-1]
        before = Game.from_moves(game.moves[:k])
        pick = game.moves[k].square
        minimax_values = minimax_solver().score_legal_moves(before.board)
        st.markdown(f"**The fly played {pick}")
        st.altair_chart(
            value_grid(agent.raw_move_scores(before), frozenset({pick}), "Fly (pick outlined)")
            | value_grid(
                minimax_values,
                best_squares(minimax_values, before.board.legal_mask()),
                "Minimax (best outlined)",
            )
        )
        st.caption(
            "Each square shows the predicted value of O playing there: +1 is a "
            "win for the fly, −1 a loss, 0 a draw.")


def render_brain(
    game: Game,
    agent: FlyAgent,
    steps: np.ndarray,
    order: np.ndarray,
    bands: list[tuple[str, int, int]],
) -> None:
    """Activity over time, per-family activity, wiring and eigenvalues."""
    config = agent.reservoir.config
    connectome = agent.reservoir.connectome

    st.markdown("**Activity over time**")
    if game.moves:
        figure = raster_figure(steps, order, bands, game, config.steps_per_move)
        st.pyplot(figure)
        plt.close(figure)
        st.caption(
            "Each row is a neuron, grouped by cell family; each column a time step. "
            "Red is positive firing, blue negative. Every move is held on the input "
            f"for {config.steps_per_move} steps, then the next one arrives.")
        # Mean |x| per family over time: which populations respond, and how fast it fades
        by_family = pd.DataFrame(
            {name: np.abs(steps[:, order[start:stop]]).mean(axis=1) for name, start, stop in bands}
        )
        st.line_chart(by_family, x_label="time step", y_label="mean |x|")
    else:
        st.info("Make a move to see the network respond.")

    wiring_column, spectrum_column = st.columns(2)
    with wiring_column:
        st.markdown("**Wiring**")
        figure = connectivity_figure(connectome, order, bands)
        st.pyplot(figure)
        plt.close(figure)
        st.caption(
            "Synapse counts (log scale), rows receiving and columns sending. "
            "Blue columns are inhibitory neurons. Tick \"Scramble wiring\" to see the control.")
    with spectrum_column:
        st.markdown("**Eigenvalues**")
        st.altair_chart(eigenvalue_chart(agent.reservoir))
        st.caption(
            "The recurrent matrix is rescaled so the outermost eigenvalue sits on the "
            "spectral-radius circle. The operator norm (the most one step can stretch a "
            "state) is larger; below 1 would guarantee that echoes fade.")

    if connectome.cell_types is None:
        st.caption("No cell types on disk, so neurons aren't grouped by family.")


def render_experiments(
    config: ReservoirConfig,
    connectome_key: tuple,
    orderings: int,
    data_seed: int,
    network: NetworkStats,
    report: TrainingReport,
) -> None:
    """Controls vs the fly, the memory curve, and the network's fixed properties."""
    connectome_seed = connectome_key[2]
    experiment_key = (config, orderings, data_seed, connectome_key)
    if st.button("Run comparison", type="primary",
                 help="Trains the controls at the current dynamics settings (a few minutes)."):
        st.session_state.experiment_key = experiment_key

    # Only show results computed for the current settings
    if st.session_state.get("experiment_key") == experiment_key:
        comparison = compare_variants(config, orderings, data_seed, connectome_seed)
        st.markdown("**Does the wiring matter?**")
        st.altair_chart(
            alt.Chart(comparison).mark_bar().encode(
                x=alt.X("optimal moves (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("variant:N", sort=None, title=None),
                yOffset="set:N",
                color=alt.Color("set:N", title=None),
                tooltip=["variant", "set", alt.Tooltip("optimal moves (%):Q", format=".1f")],
            ).properties(height=260),
            width="stretch",
        )
        st.caption(
            "All variants use the current dynamics settings and the same train/test split. "
            "\"No brain\" readouts see the board directly instead of the network state.")

        st.markdown("**Memory: can past moves be read back from the final state?**")
        st.line_chart(memory_curve(connectome_key, config, orderings, data_seed))
        st.caption(
            "Held-out R² of a linear readout recovering the move made k moves ago. "
            "1 means remembered perfectly, 0 means forgotten.")
    else:
        st.info("Press **Run comparison** to train the controls and measure memory at these settings.")

    st.markdown("**Network details**")
    st.dataframe(
        pd.DataFrame({
            "value": [
                f"{network.n_neurons}", f"{network.density:.1%}",
                f"{network.inhibitory_fraction:.0%}",
                f"{network.orphan_neurons} (no direct board input)",
                f"{network.raw_spectral_radius:.1f} → {network.spectral_radius:.2f}",
                f"{network.operator_norm:.2f}",
                "yes" if network.contraction_guaranteed else "no",
                f"{report.n_games} games over {report.n_positions} positions",
                f"{np.median(report.chosen_alphas):.2g}",
            ]
        }, index=[
            "Neurons", "Density", "Inhibitory", "Orphan neurons", "Spectral radius (raw → scaled)",
            "Operator norm", "Contraction guaranteed", "Training set", "Ridge α (median)",
        ]),
        width="stretch",
    )