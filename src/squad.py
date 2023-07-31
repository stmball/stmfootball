"""Class object for managing a squad of players."""
import typing as tp
from collections import Counter

import pandas as pd

from src.core import FootballError
from src.player import Player, Position


class SquadError(FootballError):
    """Base class for exceptions in this module."""

    pass


class Squad:
    """Class object for managing a squad of players."""

    def __init__(
        self,
        squad: tp.List[Player],
        team: tp.List[Player],
        captain: Player,
        vice_captain: Player,
    ) -> None:
        """Initialise a Squad object.

        Args:
            squad (tp.List[Player]): List of players in the squad
            team (tp.List[Player]): List of currently active team
            captain (Player): Captain of the team
            vice_captain (Player): Vice captain of the team
        """
        self.squad = squad
        self.team = team
        self.captain = captain
        self.vice_captain = vice_captain

    @property
    def squad(self) -> tp.List[Player]:
        """Get the squad of players."""
        return self._squad

    @squad.setter
    def squad(self, squad: tp.List[Player]) -> None:
        """Set the squad of players."""
        if not isinstance(squad, list) or not all(
            isinstance(player, Player) for player in squad
        ):
            raise SquadError("Squad must be a list of players")

        elif len(squad) != 15:
            raise SquadError(f"Squad must contain 15 players, not {len(squad)}")

        elif not self._check_squad_positions(squad):
            raise SquadError("Squad is not valid")

        elif not self._check_squad_cost(squad):
            raise SquadError("Squad is too expensive")

        elif not self._check_squad_max_3_from_same_team(squad):
            raise SquadError("Squad has too many players from the same team")

        else:
            self._squad = squad

    @property
    def team(self) -> tp.List[Player]:
        """Get the team of players."""
        return self._team

    @team.setter
    def team(self, team: tp.List[Player]) -> None:
        """Set the team of players."""
        if not isinstance(team, list) or not all(
            isinstance(player, Player) for player in team
        ):
            raise SquadError("Team must be a list of players")

        elif not all(player in self.squad for player in team):
            raise SquadError("Team must be a subset of the squad")

        elif len(team) != 11:
            raise SquadError("Team must contain 11 players")

        elif not self._check_team_positions(team):
            raise SquadError("Team is not valid")

        else:
            self._team = team

    @property
    def captain(self) -> Player:
        """Get the captain of the team."""
        return self._captain

    @captain.setter
    def captain(self, captain: Player) -> None:
        """Set the captain of the team."""
        if not isinstance(captain, Player):
            raise SquadError("Captain must be a player")
        elif captain not in self._team:
            raise SquadError("Captain must be in the team")
        else:
            self._captain = captain

    @property
    def vice_captain(self) -> Player:
        """Get the vice captain of the team."""
        return self._vice_captain

    @vice_captain.setter
    def vice_captain(self, vice_captain: Player) -> None:
        """Set the vice captain of the team."""
        if not isinstance(vice_captain, Player):
            raise SquadError("Vice captain must be a player")
        elif vice_captain not in self._team:
            raise SquadError("Vice captain must be in the team")
        elif vice_captain.name == self.captain.name:
            raise SquadError("Vice captain must be different to the captain")
        else:
            self._vice_captain = vice_captain

    def squad_cost(self) -> int:
        """Get the cost of the squad."""
        return sum([player.cost for player in self.squad])

    def team_cost(self) -> int:
        """Get the cost of the team."""
        return sum([player.cost for player in self.team])

    def squad_total_points(
        self, points_df: pd.DataFrame, points_col: str = "total_points"
    ) -> int:
        """Calculate the total points of a squad from a dataframe of points."""
        points_df["full name"] = (
            points_df["first_name"] + " " + points_df["second_name"]
        )
        only_squad = points_df[
            points_df["full name"].isin([player.name for player in self.squad])
        ]
        return sum(only_squad[points_col])

    def team_total_points(
        self, points_df: pd.DataFrame, points_col: str = "total_points"
    ) -> int:
        """Calculate the total points of a squad from a dataframe of points."""
        points_df["full name"] = (
            points_df["first_name"] + " " + points_df["second_name"]
        )
        only_team = points_df[
            points_df["full name"].isin([player.name for player in self.team])
        ]
        return sum(only_team[points_col])

    def _check_squad_positions(self, squad: tp.List[Player]) -> bool:
        """Check the positions of the players in the squad.

        Args:
            squad (tp.List[Player]): List of players in the squad

        Returns:
            bool: If the team meets the position requirements
        """
        positions = [player.position for player in squad]
        counts = Counter(positions)
        return (
            counts[Position.GK] == 2
            and counts[Position.DEF] == 5
            and counts[Position.MID] == 5
            and counts[Position.FWD] == 3
        )

    def _check_team_positions(self, team: tp.List[Player]) -> bool:
        """Check the positions of the players in the team.

        Args:
            team (tp.List[Player]): List of players in the team

        Returns:
            bool: If the team meets the position requirements
        """
        positions = [player.position for player in team]
        counts = Counter(positions)
        return (
            counts[Position.GK] == 1
            and counts[Position.DEF] >= 3
            and counts[Position.FWD] >= 1
        )

    def _check_squad_cost(self, squad: tp.List[Player]) -> bool:
        """Check the team cost is under the limit.

        Args:
            squad (tp.List[Player]): List of players in the squad

        Returns:
            bool: If the team meets the cost requirements
        """
        return sum([player.cost for player in squad]) <= 1000

    def _check_squad_max_3_from_same_team(self, squad: tp.List[Player]) -> bool:
        """Check if the squad has no more than 3 players from the same team.

        Args:
            squad (tp.List[Player]): List of players in the squad

        Returns:
            bool: If the team meets the common team requirements
        """
        teams = [player.team for player in squad]
        counts = Counter(teams)
        return all(count <= 3 for count in counts.values())

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame):
        """Create a squad from a pandas DataFrame."""
        squad = [Player.from_pandas_row(row) for _, row in df.iterrows()]

        # TODO: Make this more robust - for now do a 4-4-2
        gk = [player for player in squad if player.position == Position.GK][0]
        defenders = [player for player in squad if player.position == Position.DEF][:4]
        midfielders = [player for player in squad if player.position == Position.MID][
            :4
        ]
        forwards = [player for player in squad if player.position == Position.FWD][:2]
        team = [gk] + defenders + midfielders + forwards
        captain = team[0]
        vice_captain = team[1]

        return cls(squad, team, captain, vice_captain)

    @classmethod
    def from_player_list(cls, player_list: tp.List[Player]):
        """Create a squad from a list of players, with random team, captain and vice captain."""
        squad = player_list
        gk = [player for player in squad if player.position == Position.GK][0]
        defenders = [player for player in squad if player.position == Position.DEF][:4]
        midfielders = [player for player in squad if player.position == Position.MID][
            :4
        ]
        forwards = [player for player in squad if player.position == Position.FWD][:2]
        team = [gk] + defenders + midfielders + forwards
        captain = team[0]
        vice_captain = team[1]

        return cls(squad, team, captain, vice_captain)
