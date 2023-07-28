"""Module for the Player class."""

from enum import Enum

import pandas as pd

from src.core import FootballError


class PlayerError(FootballError):
    """Base class for exceptions in this module."""

    pass


class Position(Enum):
    """Enum for player positions."""

    GK = 1
    DEF = 2
    MID = 3
    FWD = 4


class Player:
    """Class object for a player in the game."""

    def __init__(self, name: str, position: Position, cost: int, team: str) -> None:
        """Initialise a Player object.

        Args:
            name (str): Name of the player (first name, last name)
            position (Position): Position of the player
            cost (int): Cost of the player (in 100ks (???))
            team (str): Team of the player
        """
        self.name = name
        self.position = position
        self.cost = cost
        self.team = team

    def __repr__(self) -> str:
        """Return a string representation of the Player object."""
        return self.name

    @classmethod
    def from_pandas_row(cls, row: pd.Series):
        """Create a Player from a pandas Series."""
        position = Position(row["element_type"])
        cost = row["now_cost"]
        team = row["team"]
        return cls(f'{row["first_name"]} {row["second_name"]}', position, cost, team)
