"""Streamlit app to play tic-tac-toe against the fly and look inside its brain.

Run: uv run streamlit run tic_tac_fly/app.py
"""

from __future__ import annotations

import streamlit as st

from tic_tac_fly.agent import FlyAgent
from tic_tac_fly.brain.diagnostics import NetworkStats, NetworkStateStats
from tic_tac_fly.brain.reservoir import ReservoirConfig
from tic_tac_fly.game.board import Game, Player
from tic_tac_fly.ui.cached import board_baseline, build_agent, load_connectome
from tic_tac_fly.ui.charts import neuron_groups
from tic_tac_fly.ui.views import render_brain, render_experiments, render_play

st.set_page_config(page_title="tic-tac-fly", layout="wide")

# ---------------------------------------------------------------- settings

sidebar = st.sidebar
sidebar.title("Settings")

with sidebar.form("settings"):
    # Widgets in a form only send their values when the button is pressed
    st.subheader("Dynamics")
    config = ReservoirConfig(
        spectral_radius=st.slider(
            "Spectral radius", 0.10, 1.50, 0.95, 0.05,
            help="Loop gain. Lower and the network forgets (desaturates). Higher and it saturates."),
        leak_rate=st.slider(
            "Leak rate", 0.05, 1.00, 0.35, 0.05,
            help="Delta-t over the membrane time constant. 1.0 means no memory. Lower means knowledge leaks less."),
        steps_per_move=st.slider("Simulation steps per move", 1, 12, 4),
        input_density=st.slider("Afferent density", 0.05, 1.00, 0.15, 0.05),
        input_scale=st.slider("Afferent strength", 0.1, 3.0, 1.0, 0.1),
        seed=int(st.number_input("Network seed", 0, 9999, 42)),
    )

    st.subheader("Connectome")
    use_predicted = st.checkbox(
        "Use predicted transmitter signs from dataset", value=True,
        help="Uncheck to assign signs randomly (still one sign per neuron).")
    inhibitory_fraction = st.slider(
        "Random inhibitory fraction", 0.0, 1.0, 0.70, 0.05,
        help="Only used when predicted signs are off. Sets what fraction of neurons should be inhibitory (-1.0).")
    scramble = st.checkbox(
        "Scramble wiring", value=False,
        help="Shuffle the synapse counts, scrambling the neural wiring of the connectome, keeping density and weight spread. "
             "The control for whether the connectome's structure matters.")
    connectome_seed = int(st.number_input(
        "Connectome seed", 0, 9999, 42, help="Seeds random signs and scrambling."))

    st.subheader("Training")
    orderings = st.slider(
        "Move orders per position", 1, 12, 4,
        help="How many different paths to the same position the readout sees. "
             "More is slower to train but means the model generalises better accross board histories.")
    data_seed = int(st.number_input("Data seed", 0, 9999, 0))

    st.form_submit_button("Retrain", type="primary", width="stretch")

if use_predicted and not load_connectome().predicted_signs:
    sidebar.warning(
        "No predicted signs on disk! Signs are set to random. Delete "
        "flywire_eb_adjacency.npy and restart the server to download them.")

connectome_key = (-1.0 if use_predicted else inhibitory_fraction, scramble, connectome_seed)
agent, report = build_agent(connectome_key, config, orderings, data_seed)
network = NetworkStats.of(agent.reservoir)
order, bands = neuron_groups(agent.reservoir.connectome)

# ---------------------------------------------------------------- game state


def reset() -> None:
    """New game."""
    st.session_state.game = Game()


if "game" not in st.session_state:
    reset()


def next_game(game: Game, square: int, agent: FlyAgent) -> Game:
    """The position after our move and the fly's reply."""
    if game.board.is_terminal or game.board[square] != 0:
        return game
    played = game.play(square, Player.X)
    return played if played.board.is_terminal else agent.play(played)


def handle_click(square: int) -> None:
    """Button callback; runs before the script reruns."""
    st.session_state.game = next_game(st.session_state.game, square, agent)


game: Game = st.session_state.game
steps = agent.reservoir.trace_steps(game)
activity = NetworkStateStats.of(steps[-1])

# ---------------------------------------------------------------- header

st.title("tic-tac-fly")

_, baseline = board_baseline(orderings, data_seed, pairs=False)
top = st.columns(5)
top[0].metric("Neurons", network.n_neurons)
top[1].metric("Inhibitory", f"{network.inhibitory_fraction:.0%}")
top[2].metric(
    "Optimal moves (held out)",
    f"{report.held_out_play_rate:.1f}%" if report.held_out_play_rate is not None else "n/a",
    delta=(
        f"{report.held_out_play_rate - baseline:+.1f} vs no brain"
        if report.held_out_play_rate is not None else None
    ),
    help="How often the fly picks a minimax-optimal move on OOD positions."
         f"On training positions: {report.optimal_play_rate:.1f}%. "
         f"A linear readout of the raw board scores {baseline:.1f}%.")
top[3].metric(
    "Quiet neurons", f"{activity.desaturated:.0%}" if game.moves else "–",
    help="Neurons with |x| < 0.05 right now. They carry little information.")
top[4].metric(
    "Saturated", f"{activity.saturated:.0%}" if game.moves else "–",
    help="Neurons pinned near ±1, which respond monotonically to any board state.")

# ---------------------------------------------------------------- tabs

play_tab, brain_tab, experiments_tab = st.tabs(["Play", "Brain", "Experiments"])
with play_tab:
    render_play(game, agent, on_square=handle_click, on_reset=reset)
with brain_tab:
    render_brain(game, agent, steps, order, bands)
with experiments_tab:
    render_experiments(config, connectome_key, orderings, data_seed, network, report)