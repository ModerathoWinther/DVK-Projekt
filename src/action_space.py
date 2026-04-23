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


SL_LEVELS = [0.001, 0.003, 0.006]
TP_LEVELS = [0.002, 0.006, 0.012]
HOLD_ACTION = Action(Direction.HOLD, None, None, 0)

i = 1
ACTION_SPACE = [HOLD_ACTION]
for direction, sl, tp in product([Direction.BUY, Direction.SELL], SL_LEVELS, TP_LEVELS):
    ACTION_SPACE.append(Action(direction, sl, tp, i))
    i += 1

UNIT_TEST_ACTION_SPACE = [HOLD_ACTION, Action(Direction.BUY, 0.01, 0.02, 1), Action(Direction.SELL, 0.01, 0.02, 2)]