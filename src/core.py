"""Core module for the football package."""

from enum import Enum


class FootballError(Exception):
    """Base class for exceptions in this module."""

    pass


class Season(Enum):
    """Enum for seasons."""

    S2016_17 = 1
    S2017_18 = 2
    S2018_19 = 3
    S2019_20 = 4
    S2020_21 = 5
    S2021_22 = 6
    S2022_23 = 7
    S2023_24 = 8

    def __repr__(self) -> str:
        """Return a string representation of the Season object."""
        return self.name.replace("_", "/")[1:]
