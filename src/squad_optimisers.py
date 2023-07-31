"""Module for ways to optimise picking a squad."""

import typing as tp
from abc import abstractmethod  # pylint: disable=E0611

import pandas as pd

from src.player import Player, Position


class BaseOptimiser:
    """Base class with common functionality for all optimisers."""

    def __init__(
        self,
        name: str = "Unnamed optimiser",
        cost_col: str = "now_cost",
        points_col: str = "predicted_points",
        formation: tp.Optional[tp.Dict[Position, int]] = None,
    ) -> None:
        """Initialise the base optimiser.

        Args:
            name (str, optional): Name for the optimiser. Defaults to "Unnamed optimiser".
            cost_col (str, optional): Column name for the cost. Defaults to "now_cost".
            points_col (str, optional): Column name for the points. Defaults to "predicted_points".
            formation (tp.Optional[tp.Dict[Position, int]], optional): Formation to try and fit a team to. Defaults to None.
        """
        self.budget = 1000
        self.squad: tp.List[Player] = []
        self.team: tp.List[Player] = []
        self.name = name
        self.cost_col = cost_col
        self.points_col = points_col
        self.formation = (
            formation
            if formation
            else {Position.GK: 1, Position.DEF: 3, Position.MID: 4, Position.FWD: 3}
        )
        self.squad_numbers = {
            Position.GK: 2,
            Position.DEF: 5,
            Position.MID: 5,
            Position.FWD: 3,
        }

    @abstractmethod  # pylint: disable=E1101
    def optimise(self, df) -> tp.List[Player]:
        """Optimise a squad according to some algorithm."""
        pass

    def _player_team_squad_rule(self, player: Player) -> bool:
        """Check if a player can fit in the squad given the max 3 players from a team rule.

        Args:
            player (Player): Candidate player to add to squad.

        Returns:
            bool: If the player can fit in the squad.
        """
        return len([p for p in self.squad if p.team == player.team]) < 3

    def _squad_position_rule(self, player):
        spaces_available = self.squad_numbers[player.position] - len(
            [p for p in self.squad if p.position == player.position]
        )
        return spaces_available > 0

    def _get_cheap_players(self, df) -> tp.List[Player]:
        players: tp.List[Player] = []
        for position in Position:
            number_to_add = self.squad_numbers[position] - self.formation[position]
            subset = (
                df[df["element_type"] == position.value]
                .sort_values(by=self.cost_col, ascending=True)
                .head(number_to_add)
            )
            self.squad.extend(
                [Player.from_pandas_row(row) for _, row in subset.iterrows()]
            )

        for player in players:
            self.budget -= player.cost

        return players


class Efficient(BaseOptimiser):
    """Efficient optimiser.

    The Efficient optimiser works by greedily adding players to the squad based on their cost/point ratio.
    """

    def __init__(
        self,
        cost_col: str,
        points_col: str,
        formation: tp.Optional[tp.Dict[Position, int]] = None,
        budget_breakdown: tp.Dict[Position, float] = {
            Position.GK: 250,
            Position.DEF: 250,
            Position.MID: 250,
            Position.FWD: 250,
        },
    ) -> None:
        """Initialise the Efficient optimiser.

        Args:
            cost_col (str): Cost column in input dataframe.
            points_col (str): Points column in input dataframe.
            formation (tp.Optional[tp.Dict[Position, int]], optional): Formation to try and fit to. Defaults to None.
            budget_breakdown (tp.Dict[Position, float]): Budget for each position. Defaults to { Position.GK: 250, Position.DEF: 250, Position.MID: 250, Position.FWD: 250, }.
        """
        super().__init__(
            name="Efficient",
            cost_col=cost_col,
            points_col=points_col,
            formation=formation,
        )

        self.budget_breakdown = budget_breakdown

    def optimise(
        self,
        df: pd.DataFrame,
    ) -> tp.List[Player]:
        """Optimise a squad according to the Efficient algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats
            budget_breakdown (tp.Dict[Position, float]): Budget allocated to each position

        Returns:
            tp.List[Player]: List of players in the squad
        """
        df["value"] = df[self.points_col] / df[self.cost_col]
        wiggle_budget = 0.0

        for position, budget in self.budget_breakdown.items():
            num_to_add = self.squad_numbers[position]

            new_players, wiggle_budget = self.greedy_add(
                df, position, budget + wiggle_budget, num_to_add
            )

            self.squad.extend(new_players)

        return self.squad

    def greedy_add(
        self,
        df: pd.DataFrame,
        position: Position,
        pos_budget: float,
        num_to_add: int,
    ) -> tp.Tuple[tp.List[Player], float]:
        """Add players to the squad greedily, starting with the best value for money players.

        Args:
            df (pd.DataFrame): Data frame of all players with their stats
            position (Position): Position to add players to
            pos_budget (float): Budget for the position
            num_to_add (int): Number of players to add to the squad from the position

        Returns:
            tp.Tuple[tp.List[Player], float]: List of players added to the squad, and the budget left
        """
        subset = df[df["element_type"] == position.value].sort_values(
            by="value", ascending=False
        )

        pos_squad: tp.List[Player] = []

        while len(pos_squad) < num_to_add:
            player = subset.iloc[0]
            if self._player_team_squad_rule(player):
                pos_squad.append(Player.from_pandas_row(player))
                pos_budget -= player["now_cost"]
                subset = subset.drop(player.name)
        return pos_squad, pos_budget


class Efficientv2(BaseOptimiser):
    """Version 2 of the efficient algorithm above.

    In this version, we first fill the non-team positions in the club with the cheapest players.
    We then fill the team positions with the best value for money players, until the correct number of players in the squad is reached.
    """

    def __init__(
        self,
        cost_col: str,
        points_col: str,
        formation: tp.Optional[tp.Dict[Position, int]] = None,
        budget_breakdown: tp.Dict[Position, float] = {
            Position.GK: 250,
            Position.DEF: 250,
            Position.MID: 250,
            Position.FWD: 250,
        },
    ) -> None:
        """Initialise an Efficientv2 optimiser.

        Args:
            cost_col (str): Cost column in input dataframe.
            points_col (str): Score column in input dataframe.
            budget_breakdown (tp.Dict[Position, float]): Budget for each position
            formation (tp.Dict[Position, int]): Number of players in each position
        """
        super().__init__(
            name="Efficientv2",
            cost_col=cost_col,
            points_col=points_col,
            formation=formation,
        )

        self.budget_breakdown = budget_breakdown

        if sum(self.budget_breakdown.values()) != self.budget:
            raise ValueError(f"Budget breakdown doesn't sum to {self.budget}")

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Optimise a squad according to the Efficientv2 algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        self.df = df
        self.squad = self._get_cheap_players(self.df)

        self.forward_pass()
        self.backward_pass()

        return self.squad

    def forward_pass(self) -> "Efficientv2":
        """Try and fill the squad with the best value for money players.

        Raises:
            ValueError: If the budget is negative

        Returns:
            Efficientv2: self
        """
        # Calculate the value of each player
        self.df["value"] = self.df[self.points_col] / self.df[self.cost_col]

        for _, row in self.df.sort_values(by="value", ascending=False).iterrows():
            # Break the loop if the squad is full
            if len(self.squad) == 15:
                break

            # Distribute the position budget to the other positions if the position is full
            self.redistribute_budget()

            # Check for negative budget
            if self.budget <= 0:
                raise ValueError("Budget is negative")

            player = Player.from_pandas_row(row)

            if (
                self._squad_position_rule(player)
                and self.check_position_budget(player)
                and self._player_team_squad_rule(player)
            ):
                self.squad.append(player)
                self.team.append(player)
                self.budget -= player.cost
                self.budget_breakdown[player.position] -= player.cost

        return self

    def backward_pass(self) -> "Efficientv2":
        """After the forward pass, go back through the squad and replace the lowest value players with higher value players with more points.

        Returns:
            Efficientv2: self
        """
        # Get the players in the team
        self.df["full name"] = self.df["first_name"] + " " + self.df["second_name"]
        only_team = self.df[self.df["full name"].isin([p.name for p in self.team])]

        for _, row in only_team.sort_values(
            by=self.points_col, ascending=True
        ).iterrows():
            # Get the lowest performing player in the team, and get the players
            # available to replace them
            player = Player.from_pandas_row(row)
            available_players = self.df[
                (self.df["element_type"] == player.position.value)
                & (self.df[self.cost_col] < self.budget + player.cost)
            ]

            # Iterate through the available players, and replace the lowest performing
            # player with the highest performing player that fits the rules
            for _, row in available_players.sort_values(
                by=self.points_col, ascending=False
            ).iterrows():
                new_player = Player.from_pandas_row(row)
                # If the new player is a valid player, replace the old player with the
                # new player
                if (
                    self._squad_position_rule(new_player)
                    and self.check_position_budget(new_player)
                    and self._player_team_squad_rule(new_player)
                ):
                    # Replace the player in the squad and the team, and recalculate
                    # the budget
                    self.squad.remove(player)
                    self.squad.append(new_player)
                    self.team.remove(player)
                    self.team.append(new_player)
                    self.budget += player.cost
                    self.budget_breakdown[player.position] += player.cost
                    self.budget -= new_player.cost
                    self.budget_breakdown[new_player.position] -= new_player.cost
                    break

        return self

    def _get_cheap_players(self, df: pd.DataFrame) -> tp.List[Player]:
        """Get the cheapest players in each position to fill the squad. Overwrites the base class method.

        Args:
            df (pd.Dataframe): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        players: tp.List[Player] = []

        for position in Position:
            # Calculate how many non-team players there are in the squad
            number_to_add = self.squad_numbers[position] - self.formation[position]

            # Get the cheapest players in the position
            subset = (
                df[df["element_type"] == position.value]
                .sort_values(by=self.cost_col, ascending=True)
                .head(number_to_add)
            )
            self.squad.extend(
                [Player.from_pandas_row(row) for _, row in subset.iterrows()]
            )

        # Recalculate the budget breakdown
        for player in players:
            self.budget -= player.cost
            self.budget_breakdown[player.position] -= player.cost

        return players

    def check_position_budget(self, player: Player) -> bool:
        """Check if a position has enough budget to add a player.

        Args:
            player (Player): Player to add to the squad

        Returns:
            bool: If the position has enough budget to add the player
        """
        return player.cost <= self.budget_breakdown[player.position]

    def redistribute_budget(self):
        """Redistribute the budget if a position is full."""
        wiggle_room = 0
        num_to_redistribute = 4

        for position in Position:
            # Check every position - if it is full, add the remaining budget to the wiggle room
            if (
                len([p for p in self.squad if p.position == position])
                == self.squad_numbers[position]
                and self.budget_breakdown[position] > 0
            ):
                wiggle_room += self.budget_breakdown[position]
                self.budget_breakdown[position] = 0

                # And reduce the number of positions to redistribute to
                num_to_redistribute -= 1

        # Then redistribute the budget to the other positions
        if wiggle_room > 0:
            for position in Position:
                if (
                    len([p for p in self.squad if p.position == position])
                    < self.squad_numbers[position]
                ):
                    self.budget_breakdown[position] += wiggle_room / num_to_redistribute


class Average(BaseOptimiser):
    """Average optimiser that tries to pick the average player in each position."""

    def __init__(
        self,
        cost_col: str,
        points_col: str,
        formation: tp.Optional[tp.Dict[Position, int]] = None,
    ) -> None:
        """Initialise the Average optimiser.

        Args:
            cost_col (str): Cost column in input dataframe.
            points_col (str): Points column in input dataframe.
        """
        super().__init__("Average", cost_col, points_col, None)

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Optimise a squad according to the Average algorithm.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: List of players in the squad
        """
        # Calculate the average cost for each position
        avg_cost = df[self.cost_col].mean()

        # Calculate how far each player is from the average cost
        df["delta_mean"] = abs(df[self.cost_col] - avg_cost)

        # Sort the players by how far they are from the average cost
        for _, row in df.sort_values(by="delta_mean", ascending=True).iterrows():
            player = Player.from_pandas_row(row)

            # If the player is a valid player, add them to the squad
            if (
                self._squad_position_rule(player)
                and self._player_team_squad_rule(player)
                and self.budget > player.cost
            ):
                self.squad.append(player)
                self.budget -= player.cost

        return self.squad


class Random(BaseOptimiser):
    """Choose a squad randomly from the given dataframe, skipping players if they can't fit."""

    def __init__(
        self,
        cost_col: str,
        points_col: str,
    ) -> None:
        """Initialise a Random optimiser.

        Args:
            cost_col (str): Cost column in input dataframe.
            points_col (str): Points column in input dataframe.
        """
        super().__init__("Random", cost_col, points_col, None)

    def optimise(self, df: pd.DataFrame) -> tp.List[Player]:
        """Choose a squad randomly from the given dataframe, skipping players if they can't fit.

        Args:
            df (pd.DataFrame): Dataframe of all players with their stats

        Returns:
            tp.List[Player]: Players in the squad
        """
        while True:
            # Break the loop if the squad is full
            if len(self.squad) == 15:
                break

            # Otherwise, take a random player
            player = Player.from_pandas_row(df.sample().iloc[0])

            # If they are a valid player, add them to the squad
            if (
                self._squad_position_rule(player)
                and self._player_team_squad_rule(player)
                and self.budget > player.cost
            ):
                self.squad.append(player)
                self.budget -= player.cost

        return self.squad
