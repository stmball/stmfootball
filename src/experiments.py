"""Modules for wrappers around experiements."""

import typing as tp

import numpy.typing as npt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from src.core import FootballError
from src.predictors import BasePredictor


class ExperimentError(FootballError):
    """Base class for exceptions in this module."""

    pass


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
