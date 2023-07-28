"""Class object for managing a squad of players."""
import typing as tp
from collections import Counter

import pandas as pd

from core import FootballError
from player import Player, Position


class SquadError(FootballError):
    """Base class for exceptions in this module."""
    pass

class Squad:

    def __init__(self, squad: tp.List[Player], team: tp.List[Player], captain: Player, vice_captain: Player) -> None:
        self.squad = squad
        self.team = team
        self.captain = captain
        self.vice_captain = vice_captain
    
    @property
    def squad(self) -> tp.List[Player]:
        return self._squad
    
    @squad.setter
    def squad(self, squad: tp.List[Player]) -> None:

        if not isinstance(squad, list) or not all(isinstance(player, Player) for player in squad):
            raise SquadError("Squad must be a list of players")
        
        elif len(squad) != 15:
            raise SquadError("Squad must contain 15 players")

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
        return self._team
    
    @team.setter
    def team(self, team: tp.List[Player]) -> None:
    
            if not isinstance(team, list) or not all(isinstance(player, Player) for player in team):
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
        return self._captain
    
    @captain.setter
    def captain(self, captain: Player) -> None:
        if not isinstance(captain, Player):
            raise SquadError("Captain must be a player")
        elif captain not in self._team:
            raise SquadError("Captain must be in the team")
        else:
            self._captain = captain
    
    @property
    def vice_captain(self) -> Player:
        return self._vice_captain
    
    @vice_captain.setter
    def vice_captain(self, vice_captain: Player) -> None:
        if not isinstance(vice_captain, Player):
            raise SquadError("Vice captain must be a player")
        elif vice_captain not in self._team:
            raise SquadError("Vice captain must be in the team")
        else:
            self._vice_captain = vice_captain

    def _check_squad_positions(self, squad: tp.List[Player]) -> bool:
        positions = [player.position for player in squad]
        counts = Counter(positions)
        return counts[Position.GK] == 2 and counts[Position.DEF] == 5 and counts[Position.MID] == 5 and counts[Position.FWD] == 3

    def _check_team_positions(self, team: tp.List[Player]) -> bool:
        positions = [player.position for player in team]
        counts = Counter(positions)
        return counts[Position.GK] == 1 and counts[Position.DEF] >= 3 and counts[Position.FWD] >= 1
    
    def _check_squad_cost(self, squad: tp.List[Player]) -> bool:
        return sum([player.cost for player in squad]) <= 1000

    def _check_squad_max_3_from_same_team(self, squad: tp.List[Player]) -> bool:
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
        midfielders = [player for player in squad if player.position == Position.MID][:4]
        forwards = [player for player in squad if player.position == Position.FWD][:2]
        team = [gk] + defenders + midfielders + forwards
        captain = team[0]
        vice_captain = team[1]

        return cls(squad, team, captain, vice_captain)
