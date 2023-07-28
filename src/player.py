import typing as tp
from enum import Enum

import pandas as pd

from core import FootballError


class PlayerError(FootballError):
    """Base class for exceptions in this module."""
    pass

class Position(Enum):
    GK = 1
    DEF = 2
    MID = 3
    FWD = 4

class Player:

    def __init__(self, name: str, position: Position, cost: int, team: str) -> None:

        self.name = name
        self.position = position
        self.cost = cost
        self.team = team

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def from_pandas_row(cls, row: pd.Series):
        """Create a Player from a pandas Series."""

        position = Position(row['element_type'])
        cost = row['now_cost']
        team = row['team']
        return cls(f'{row["first_name"]} {row["second_name"]}', position, cost, team)
