"""Cached builders for the app: connectome, training data, agents and experiments.

Each function is wrapped in @st.cache_resource, so Streamlit computes it once
per distinct set of arguments and reuses the result across reruns and sessions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import RidgeCV

from tic_tac_fly.agent import FlyAgent, TrainingReport
from tic_tac_fly.brain.connectome import Connectome
from tic_tac_fly.brain.readout import SquareReadout
from tic_tac_fly.brain.reservoir import Reservoir, ReservoirConfig
from tic_tac_fly.game.minimax import MinimaxSolver
from tic_tac_fly.game.positions import PositionSet

TEST_FRACTION = 0.2


@st.cache_resource
def load_connectome() -> Connectome:
    """The connectome from disk (downloaded from FlyWire on first run)."""
    return Connectome.load_or_fetch()


@st.cache_resource
def build_connectome(inhibitory: float, scramble: bool, seed: int) -> Connectome:
    """The connectome with the chosen signs and optional scrambling.

    Args:
        inhibitory: fraction of random inhibitory neurons, or a negative
            value to keep the loaded (predicted) signs.
        scramble: shuffle the wiring as a control.
        seed: seeds random signs and scrambling.
    """
    connectome = load_connectome()
    if inhibitory >= 0:
        # replace() keeps root_ids and cell_types; only the signs change
        random_signs = Connectome.random_signs(connectome.synapses, inhibitory, seed).signs
        connectome = replace(connectome, signs=random_signs, predicted_signs=False)
    return connectome.scrambled(seed) if scramble else connectome


@st.cache_resource(show_spinner="Generating positions...")
def load_positions(orderings: int, seed: int) -> PositionSet:
    """All training positions, with up to `orderings` move histories each."""
    return PositionSet.generate_exhaustively(orderings_per_board=orderings, seed=seed)


@st.cache_resource(show_spinner="Training readout...")
def build_agent(
    connectome_key: tuple, config: ReservoirConfig, orderings: int, seed: int
) -> tuple[FlyAgent, TrainingReport]:
    """A trained agent and its report for these settings."""
    agent = FlyAgent(Reservoir(build_connectome(*connectome_key), config))
    report = agent.train_readout(load_positions(orderings, seed), TEST_FRACTION, seed)
    return agent, report


@st.cache_resource
def minimax_solver() -> MinimaxSolver:
    """Shared minimax solver for the decision view."""
    return MinimaxSolver()


def optimal_rate(readout: SquareReadout, positions: PositionSet, features: np.ndarray) -> float:
    """% of games where the readout's best legal square is minimax-optimal."""
    scores = readout.predict(features)
    hits = sum(
        int(np.argmax(np.where(positions.boards[i].legal_mask(), scores[k], -np.inf)))
        in positions.optimal_squares(int(i))
        for k, i in enumerate(positions.game_index)
    )
    return 100.0 * hits / len(positions)


def board_features(positions: PositionSet, pairs: bool) -> np.ndarray:
    """The raw board as features, optionally with every pair of squares."""
    x = positions.game_boards.astype(float)
    if not pairs:
        return x
    # Products of every pair of squares: (n, 9, 1) * (n, 1, 9) -> (n, 81)
    return np.hstack([x, (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)])


@st.cache_resource(show_spinner="Fitting no-brain baseline...")
def board_baseline(orderings: int, seed: int, pairs: bool) -> tuple[float, float]:
    """(training, held-out) optimal-move % of a readout that sees the board directly.

    Uses the same train/test split as the fly, so the numbers are comparable.
    """
    train, test = load_positions(orderings, seed).split(TEST_FRACTION, seed)
    readout = SquareReadout().fit(
        board_features(train, pairs), train.game_boards, train.game_minmax_targets
    )
    return (
        optimal_rate(readout, train, board_features(train, pairs)),
        optimal_rate(readout, test, board_features(test, pairs)),
    )


@st.cache_resource(show_spinner="Running comparison...")
def compare_variants(
    config: ReservoirConfig, orderings: int, seed: int, connectome_seed: int
) -> pd.DataFrame:
    """Optimal-move % of the fly, its controls and the no-brain baselines."""
    real_fraction = round(load_connectome().inhibitory_fraction, 2)
    variants = {
        "Fly wiring, predicted signs": (-1.0, False, connectome_seed),
        f"Fly wiring, random signs ({real_fraction:.0%})": (real_fraction, False, connectome_seed),
        "Scrambled wiring": (-1.0, True, connectome_seed),
    }
    rows = []
    for name, key in variants.items():
        _, report = build_agent(key, config, orderings, seed)
        rows += [
            (name, "training", report.optimal_play_rate),
            (name, "held out", report.held_out_play_rate),
        ]
    for name, pairs in (("No brain: board only", False), ("No brain: board + pairs", True)):
        train_rate, test_rate = board_baseline(orderings, seed, pairs)
        rows += [(name, "training", train_rate), (name, "held out", test_rate)]
    return pd.DataFrame(rows, columns=["variant", "set", "optimal moves (%)"])


@st.cache_resource(show_spinner="Measuring memory...")
def memory_curve(
    connectome_key: tuple, config: ReservoirConfig, orderings: int, seed: int
) -> pd.DataFrame:
    """How well each past move can be read back from the final state.

    For each lag (0 = the last move), fit a linear map from final states to
    that move's signed one-hot input on training games, and score it with
    R² on held-out games. 1 means perfectly remembered, 0 means forgotten.
    """
    agent, _ = build_agent(connectome_key, config, orderings, seed)
    train, test = load_positions(orderings, seed).split(TEST_FRACTION, seed)
    train_states = agent.reservoir.encode_all(train.games)
    test_states = agent.reservoir.encode_all(test.games)

    def lagged(positions: PositionSet, states: np.ndarray, lag: int):
        # Games long enough to have a move `lag` moves before the end
        keep = [k for k, game in enumerate(positions.games) if len(game) > lag]
        targets = np.array([positions.games[k].moves[-1 - lag].one_hot() for k in keep])
        return states[keep], targets

    rows = []
    for lag in range(7):  # O is to move, so games are at most 7 moves long
        x_train, y_train = lagged(train, train_states, lag)
        x_test, y_test = lagged(test, test_states, lag)
        if len(x_train) < 10 or len(x_test) < 2:
            break
        model = RidgeCV(alphas=np.logspace(-6, 1, 15)).fit(x_train, y_train)
        rows.append({"moves ago": lag, "R²": model.score(x_test, y_test)})
    return pd.DataFrame(rows).set_index("moves ago")