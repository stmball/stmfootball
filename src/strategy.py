"""Class object for managing the decisions made by the model.

Strategies combine predictions from a model with a squad optimiser to choose a
squad of players, and handle weekly updates to the model.
"""

from pathlib import Path
from typing import List

import pandas as pd
import rich
from rich.table import Table

from src.analysis import add_predicted_points_to_df
from src.core import Season
from src.optimisers import BaseOptimiser, MIPSquad, MIPTeam
from src.player import Position
from src.predictors import BasePredictor, LSTMPredictor
from src.squad import Squad


class Strategy:
    """Abstract base class for strategies.

    Strategies combine predictions from a model with a squad optimiser to choose a
    squad of players, and handle weekly updates to the model.
    """

    def __init__(
        self,
        squad_optimiser: BaseOptimiser = MIPSquad,
        team_optimiser: BaseOptimiser = MIPTeam,
        predictor: BasePredictor = LSTMPredictor,
    ) -> None:
        """Initialise the strategy."""
        self.name = "Version v1.0"
        self.squad_optimiser = squad_optimiser
        self.team_optimiser = team_optimiser
        self.predictor = predictor
        self.seasons = [
            Season.S2016_17,
            Season.S2017_18,
            Season.S2018_19,
            Season.S2019_20,
            Season.S2020_21,
            Season.S2021_22,
            Season.S2022_23,
        ]
        # List of players to ignore due to them being traded away, injured, etc.
        self.blacklist = [
            "Jordan Henderson",
            "Fabio Henrique Tavares",
            "Riyad Mahrez",
            "Kevin De Bruyne",
            "Harry Kane",
        ]
        self.current_player_path = Path("data/data/2023-24/players_raw.csv")

    def choose_squad(self) -> Squad:
        # Get the current players
        df = pd.read_csv(self.current_player_path)

        # Add the full name column to easily find players
        df["full name"] = df["first_name"] + " " + df["second_name"]

        # Remove players from the blacklist
        df = df[~df["full name"].isin(self.blacklist)]

        # Add the predicted points to the dataframe using the chosen predictor
        df = add_predicted_points_to_df(df, self.seasons[:-1], self.predictor)

        # Find the optimal squad
        self.squad = self.squad_optimiser().optimise(df)

        # Get a dataframe of the squad
        self.squad_df = df[df["full name"].isin([p.name for p in self.squad])]

        # Get the optimal team
        self.team = self.team_optimiser().optimise(self.squad_df)
        self.team_df = self.squad_df[
            self.squad_df["full name"].isin([p.name for p in self.team])
        ]

        # Get the best captain
        self.captain = self.team_df.sort_values(
            "predicted_points", ascending=False
        ).iloc[0]
        self.vice_captain = self.team_df.sort_values(
            "predicted_points", ascending=False
        ).iloc[1]

        return self

    def weekly_update(self):
        """Update the model with new information."""
        pass

    def print_squad_stats(self, jupyter: bool = False):
        table = Table(title="Squad")
        table.add_column("Name")
        table.add_column("Position")
        table.add_column("Predicted Points")
        table.add_column("Cost")
        table.add_column("Is in Team?")
        table.add_column("Is Captain?")
        table.add_column("Is Vice Captain?")

        for _, row in self.squad_df.sort_values(
            by=["element_type", "predicted_points"], ascending=[True, False]
        ).iterrows():
            table.add_row(
                row["full name"],
                Position(row["element_type"]).name,
                f'{row["predicted_points"]:.2f}',
                str(row["now_cost"]),
                "✔️" if row["full name"] in [p.name for p in self.team] else "",
                "✔️" if row["full name"] == self.captain["full name"] else "",
                "✔️" if row["full name"] == self.vice_captain["full name"] else "",
            )

        rich.jupyter.print(table) if jupyter else rich.print(table)

        table = Table(title="Summary Statistics")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Total Cost", str(self.squad_df["now_cost"].sum()))
        table.add_row(
            "Total Predicted Points", str(self.squad_df["predicted_points"].sum())
        )
        table.add_row(
            "Total Predicted Points (Team)", str(self.team_df["predicted_points"].sum())
        )

        rich.jupyter.print(table) if jupyter else rich.print(table)

        return True
