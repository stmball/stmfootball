"""Modules for wrappers around experiements."""

import typing as tp
from abc import ABC, abstractmethod  # pylint: disable=E0611

import pandas as pd

from .squad import Squad
from .squad_optimisers import BaseOptimiser


class BaseExperiment(ABC):  # pylint: disable=E1101
    """Abstract base class for experiments."""

    @abstractmethod  # pylint: disable=E1101
    def run(self) -> tp.Any:
        """Run the experiment."""
        pass


class SquadOptimiserExperiment(BaseExperiment):
    """Class for testing squad optimisers by giving them the true values of players and seeing how they optimise."""

    def run(  # type: ignore
        self,
        df: pd.DataFrame,
        squad_optimisers: tp.List[BaseOptimiser],
        cost_col: str = "now_cost",
        points_col: str = "predicted_points",
    ) -> pd.DataFrame:
        """Run the experiment.

        Args:
            df (pd.DataFrame): Dataframe of players
            squad_optimisers (tp.List[BaseOptimiser]): List of squad optimisers to test
            cost_col (str, optional): Cost column in the dataframe. Defaults to "now_cost".
            points_col (str, optional): Points column in the dataframe. Defaults to "predicted_points".

        Returns:
            pd.DataFrame: Dataframe of results
        """
        scores = []
        for squad_optimiser in squad_optimisers:
            squad_optimiser = squad_optimiser(cost_col=cost_col, points_col=points_col)  # type: ignore
            so_name = squad_optimiser.name
            squad = Squad.from_player_list(squad_optimiser.optimise(df))

            scores.append(
                {
                    "squad_optimiser": so_name,
                    "squad_total_points": squad.squad_total_points(df),
                    "team_total_points": squad.team_total_points(df),
                    "squad_cost": squad.squad_cost(),
                    "team_cost": squad.team_cost(),
                    "squad": sorted(squad.squad, key=lambda x: x.name),
                }
            )

        return pd.DataFrame(scores)
