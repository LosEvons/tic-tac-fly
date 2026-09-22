from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property, partial, reduce
from itertools import accumulate
from typing import Iterable

import numpy as np

from tic_tac_fly.brain.connectome import Connectome
from tic_tac_fly.game.board import BOARD_SIZE, Game
