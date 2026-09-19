# Fly-Tac-Toe
Making a fruit fly play tic-tac-toe.

## Usage
```bash
uv sync
# Create an env file, and add your flywire API key to it.
# Create an account or log in at: https://codex.flywire.ai/
# You can see your token at: https://global.daf-apis.com/auth/api/v1/user/token
# You can create your token at: https://global.daf-apis.com/auth/api/v1/create_token
touch .env
uv run streamlit run tic_tac_fly/app.py
```

## How does it work?

This is essentially an echo state network, meaning the state of the game board is projected into a higher-dimensional "reservoir", which in this case is a part of a fruit fly's connectome (its brain), it echoes through the reservoir, and we train a linear model (or multiple in this case) to interpret the state of the reservoir after the input has echoed through it. This follows a recurrencce: $X_t = (1 - a)X_{t-1}+a$ $tanh(Wx_{t-1}+W_{in}u), $ where $x_0=0$

Where:
Variable|Definition
---|---
$u\in\{-1, 0, 1\}^9$|representation of the board
$W_{bio}$|signed synapse counts
$p>0$|target spectral radius (maximum eigenvalue in $W$)
$W$|$W_{bio}$ scaled to spectral radius $p$
$W_{in}$|sparse random input weigths
$x_t$|state of the network after $t$ steps
$a\in(0,1]$|leak rate

The fruit fly brain connectome dataset gives us a matrix representation of synapse counts. Synapses are either exciting or inhibiting (hyperpolarizing or depolarizing). The dataset gives this data too, but it is not yet implemented, and I use a raw approximation of the distribution, and assign signs to the synapses at random. Dale's principle states, that a neuron releases the same neurotransmitter at all of its output synapses. I make this assumption as well in this program. The leak rate describes a loss of voltage over time in the neurons. In this program it controls how much the network "remembers" between simulation rounds. Mind you, the connectome itself doesn't learn! The linear model(s) are trained to interpret the connectome's echo.

Here's a paper about the specific type of leaky-integrator echo state network (I love paywalls): https://www.sciencedirect.com/science/article/pii/S089360800700041X