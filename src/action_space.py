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
    index: int


SL_LEVELS = [0.020, 0.040, 0.080]
TP_LEVELS = [0.040, 0.080, 0.160]
HOLD_ACTION = Action(Direction.HOLD, None, None, 0)

i = 1
ACTION_SPACE = [HOLD_ACTION]
for direction, sl, tp in product([Direction.BUY, Direction.SELL], SL_LEVELS, TP_LEVELS):
    ACTION_SPACE.append(Action(direction, sl, tp, i))
    i += 1