"""Module for ways to optimise picking a squad."""

import typing as tp

import mip
import pandas as pd

from src.player import Player, Position


class OptimiserError(Exception):
    """Base class for exceptions in this module."""

    pass


class BaseOptimiser:
    """Base class for optimisers."""

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Optimise a squad according to the algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        raise NotImplementedError


class MIPSquad(BaseOptimiser):
    """Uses integer programming to find an optimal solution."""

    def __init__(
        self,
        cost_col: str = "now_cost",
        points_col: str = "predicted_points",
    ) -> None:
        """Initialise the MIP solver.

        Args:
            cost_col (str, optional): Cost column in the dataframe. Defaults to "now_cost".
            points_col (str, optional): Points column in the dataframe. Defaults to "predicted_points".
        """

        self.cost_col = cost_col
        self.points_col = points_col
        self.budget = 1000
        self.squad_numbers = {
            Position.GK: 2,
            Position.DEF: 5,
            Position.MID: 5,
            Position.FWD: 3,
        }
        self.squad = []

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Optimise a squad according to the MIP algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        # Reset the index in case it's not already
        df.reset_index(inplace=True)

        costs = df[self.cost_col].to_list()
        values = df[self.points_col].to_list()
        positions = df["element_type"].to_list()
        teams = df["team"].to_list()
        indexes = df.index.to_list()

        # Create the model
        m = mip.Model(sense=mip.MAXIMIZE)

        # Create variables for each player
        y = [m.add_var(var_type=mip.BINARY) for i in indexes]

        # Set the objective function
        m.objective = mip.xsum(y[i] * values[i] for i in indexes)

        # Add the budget constraint
        m += mip.xsum(y[i] * costs[i] for i in indexes) <= self.budget

        # Add the position constraints
        for position in Position:
            m += (
                mip.xsum(y[i] for i in indexes if positions[i] == position.value)
                == self.squad_numbers[position]
            )

        # Add the team constraint
        for team in range(1, 21):
            m += mip.xsum(y[i] for i in indexes if teams[i] == team) <= 3

        m.optimize()

        # Add the players to the squad
        for i in indexes:
            if y[i].x >= 0.99:
                self.squad.append(Player.from_pandas_row(df.iloc[i]))

        return self.squad


class MIPTeam(BaseOptimiser):
    """Given a squad, find the optimal team."""

    def __init__(
        self,
        points_col: str = "predicted_points",
    ) -> None:
        """Initialise the MIP solver.

        Args:
            points_col (str, optional): Points column in the dataframe. Defaults to "predicted_points".
        """

        self.points_col = points_col
        self.team = []

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Optimise a squad according to the MIP algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        # Reset the index in case it's not already
        df.reset_index(inplace=True)

        values = df[self.points_col].to_list()
        positions = df["element_type"].to_list()
        indexes = df.index.to_list()

        # Create the model
        m = mip.Model(sense=mip.MAXIMIZE)

        # Create variables for each player
        y = [m.add_var(var_type=mip.BINARY) for i in indexes]

        # Set the objective function
        m.objective = mip.xsum(y[i] * values[i] for i in indexes)

        # Add the position constraints
        m += mip.xsum(y[i] for i in indexes if positions[i] == Position.GK) == 1
        m += mip.xsum(y[i] for i in indexes if positions[i] == Position.DEF) >= 3
        m += mip.xsum(y[i] for i in indexes if positions[i] == Position.FWD) >= 1

        # Add the team constraint
        m += mip.xsum(y[i] for i in indexes) == 11

        m.optimize()

        # Add the players to the squad
        for i in indexes:
            if y[i].x >= 0.99:
                self.team.append(Player.from_pandas_row(df.iloc[i]))

        return self.team
