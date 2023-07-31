"""Class object for a strategy model.

Strategies combine predictions from a model with a squad optimiser to choose a
squad of players, and handle weekly updates to the model.
"""

from abc import ABC, abstractmethod  # pylint: disable=E0611
from typing import List

from src.player import Player
from src.squad import Squad


class Strategy(ABC):  # pylint disable=E1101
    """Abstract base class for strategies.

    Strategies combine predictions from a model with a squad optimiser to choose a
    squad of players, and handle weekly updates to the model.
    """

    @abstractmethod  # pylint disable=E1101
    def choose_squad(self, all_players: List[Player]) -> Squad:
        """Choose a squad of players from the given list of all players.

        Args:
            all_players (List[Player]): List of all players available to choose

        Returns:
            Squad: Squad of players, including a starting team, captain and vice captain
        """
        pass

    @abstractmethod
    def weekly_update(self):
        """Update the model with new information."""
        pass
