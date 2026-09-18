import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from tic_tac_fly.reservoir import FlyReservoir, train_minmax_readout
from tic_tac_fly.minmax import check_is_terminal, check_winner, check_draw

st.set_page_config(page_title="Game", layout="wide")
st.title("Game")

@st.cache_resource
def get_model():
    res = FlyReservoir()
    clf = train_minmax_readout(res)
    return res, clf

reservoir, readout_model = get_model()

st.sidebar.header("Variables")
leak_rate = st.sidebar.slider("Leak rate", 0.05, 1.0, 0.35, 0.05)
spectral_radius = st.sidebar.slider("Spectral radius", 0.1, 1.5, 0.95, 0.05)
simulation_steps = st.sidebar.slider("Simulation steps", 5, 60, 25, 5)

if "board" not in st.session_state:
    st.session_state.board = np.zeros(9, dtype=int)

def handle_click(index: int):
    if check_is_terminal(st.session_state.board):
        return
    if st.session_state.board[index] == 0:
        st.session_state.board[index] = 1
        if check_is_terminal(st.session_state.board):
            return
        state = reservoir.simulate(
            st.session_state.board,
            spectral_radius,
            leak_rate,
            simulation_steps
        )
        scores = readout_model.predict(state.reshape(1, -1))[0]
        scores[st.session_state.board != 0]= -np.inf
        fly_brain_move = np.argmax(scores)
        st.session_state.board[fly_brain_move] = -1

c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("You: X | Fly: 0")
    symbols = {0: " ", 1: "X", -1: "O"}
    for row in range(3):
        columns = st.columns(3)
        for c in range(3):
            index = row * 3 + c
            columns[c].button(
                symbols[st.session_state.board[index]],
                key=f"sq_{index}",
                on_click=handle_click,
                args=(index,),
                use_container_width=True
            )
    winner = check_winner(st.session_state.board)
    if winner == 1:
        st.success("You win")
    elif winner == -1:
        st.error("Fly win")
    elif check_draw(st.session_state.board):
        st.info("Draw")
            
    if st.button("Reset"):
        st.session_state.board = np.zeros(9, dtype=int)
        st.rerun()

with c2:
    st.subheader("Reservoir firing state")
    current_state = reservoir.simulate(
        st.session_state.board,
        spectral_radius,
        leak_rate,
        simulation_steps
    )
    rows = 10
    cols = -(-reservoir.n // rows)  # ceil division so all neurons fit for visualization
    padded_state = np.zeros(rows * cols)
    padded_state[:reservoir.n] = current_state

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(
        padded_state.reshape(rows, cols),
        cmap="viridis",
        aspect="auto",
        vmin=-1,
        vmax=1
    )
    ax.axis("off")
    st.pyplot(fig)
    scores = readout_model.predict(current_state.reshape(1, -1))[0]
    st.caption("Move preference")
    st.bar_chart(scores)