"""Modules for wrappers around experiements."""

import typing as tp

import numpy.typing as npt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from .predictors import BasePredictor
from .squad import Squad
from .squad_optimisers import BaseOptimiser


class SquadOptimiserExperiment:
    """Class for testing squad optimisers by giving them the true values of players and seeing how they optimise."""

    def run(
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


class PredictorExperiment:
    """Experiment for comparing predictors."""

    def run(
        self,
        historic_data: npt.NDArray,
        current_data: npt.NDArray,
        predictors: tp.List[BasePredictor],
    ) -> pd.DataFrame:
        """Run the experiment.

        Args:
            historic_data (npt.NDArray): Data for up to (but not including) the current gameweek (i.e. the training data
            current_data (npt.NDArray): The current game week (i.e. the test data)
            predictors (tp.List[BasePredictor]): Predictors to test

        Returns:
            pd.DataFrame: Sumary dataframe of results
        """
        scores = []

        for predictor in predictors:
            print("Running predictor:", predictor().name)  # type: ignore
            exp_predictor = predictor()  # type: ignore

            if exp_predictor.needs_training:
                exp_predictor.train(historic_data)

            predictions = exp_predictor.predict(historic_data)
            exp_name = exp_predictor.name

            scores.append(
                {
                    "predictor": exp_name,
                    "mse": mse(current_data, predictions),
                }
            )

        return pd.DataFrame(scores)
