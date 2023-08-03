"""Module for the Player class."""

import typing as tp
from enum import Enum
from pathlib import Path

import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression

from src.core import FootballError, Season


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
    """Class object for a current active player in the game."""

    def __init__(
        self,
        first_name: str,
        last_name: str,
        position: Position,
        cost: int,
        team: str,
    ) -> None:
        """Initialise a Player object.

        Args:
            name (str): Name of the player (first name, last name)
            position (Position): Position of the player
            cost (int): Cost of the player (in 100ks (???))
            team (str): Team of the player
        """
        self.first_name = first_name
        self.last_name = last_name
        self.name = f"{first_name} {last_name}"
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
        return cls(
            row["first_name"],
            row["second_name"],
            position,
            row["now_cost"],
            row["team"],
        )

    def predict_points_for_next_season(
        self,
        seasons: tp.List[Season],
        model: sk.base.BaseEstimator = LinearRegression(),
    ) -> float:
        """Predict the number of points the player will score in the next season.

        Args:
            seasons (tp.List[Season]): Seasons to use for prediction

        Returns:
            float: Predicted number of points
        """
        historical_data = self.get_historic_points_by_season(seasons)
        if len(list(filter(lambda x: x != 0, historical_data.values()))) < 3:
            # Not enough data to make a prediction
            return 0
        X = [[season.value] for season in seasons]
        y = list(historical_data.values())
        model.fit(X, y)
        return model.predict([[max(season.value for season in seasons) + 1]])[0]

    def get_historic_points_by_season(
        self, seasons: tp.List[Season]
    ) -> tp.Dict[Season, float]:
        """Get the total number of points the player had in each season they played.

        Args:
            seasons (tp.List[Season]): Seasons to get points for

        Returns:
            tp.Dict[Season, int]: Pairs of seasons as well as the number of points the player had in that season
        """
        points_by_week = self.get_historic_points_by_week(seasons)
        points_by_season = {a: sum(b) for a, b in points_by_week.items()}
        return points_by_season

    def get_historic_points_by_week(
        self, seasons: tp.List[Season]
    ) -> tp.Dict[Season, tp.List[float]]:
        """Get the historic points for each week in each season for a player.

        Args:
            seasons (tp.List[Season]): Seasons to get points for

        Returns:
            tp.Dict[Season, tp.List[float]]: Dictoinary of seasons and the points the player had in each week of that season
        """
        point_history = {}
        for season in seasons:
            point_history[season] = self.get_historic_points_for_season(season)

        return point_history

    def get_historic_points_for_season(self, season: Season) -> tp.List[float]:
        """Try to get the histoical points for each season for a player.

        Args:
            season (Season): Season to get points for

        Returns:
            tp.List[float]: Number of points the player had in the season. If the player did not play in the season, returns [0]
        """
        try:
            total_path = self._get_player_data_path(season)
            player_data = pd.read_csv(total_path)
            return player_data["total_points"].tolist()
        except PlayerError:
            return [0]

    def _get_player_id_for_season(self, season: Season) -> int:
        """Get the player ID for the given season."""
        base_path = Path(__file__).parent.parent
        season_folder = f"{season.name[1:].replace('_', '-')}"
        player_data = pd.read_csv(
            base_path / "data" / "data" / season_folder / "player_idlist.csv"
        )
        try:
            return player_data.loc[
                (player_data["first_name"] == self.first_name)
                & (player_data["second_name"] == self.last_name)
            ]["id"].item()
        except ValueError:
            raise PlayerError(
                f"No player with name {self.first_name} {self.last_name} found for season {season.name}"
            )

    def _get_player_data_path(self, season: Season) -> Path:
        """Get the path to the player data for the given season."""
        base_path = Path(__file__).parent.parent
        season_folder = f"{season.name[1:].replace('_', '-')}"
        player_id = self._get_player_id_for_season(season)
        if season.value < 3:
            # Seasons 2016/17 and 2017/18 have different file structure
            return (
                base_path
                / "data"
                / "data"
                / season_folder
                / "players"
                / f"{self.first_name}_{self.last_name}"
                / "gw.csv"
            )
        else:
            return (
                base_path
                / "data"
                / "data"
                / season_folder
                / "players"
                / f"{self.first_name}_{self.last_name}_{player_id}"
                / "gw.csv"
            )
