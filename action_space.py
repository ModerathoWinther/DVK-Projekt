from dataclasses import dataclass
from enum import Enum
from itertools import product


class Direction(Enum):
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class Action:
    direction: Direction
    sl: float | None
    tp: float | None


SL_LEVELS = [0.005, 0.010, 0.020]
TP_LEVELS = [0.010, 0.020, 0.040]
ACTION_SPACE = [Action(Direction.HOLD, None, None)]
for direction, sl, tp in product([Direction.BUY, Direction.SELL], SL_LEVELS, TP_LEVELS):
    ACTION_SPACE.append(Action(Direction.SELL, sl, tp))