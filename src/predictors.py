"""Module for classes responsible for the prediction of future points for a player."""
import typing as tp
import warnings
from abc import ABC, abstractmethod  # pylint: disable=E0611

import keras
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


class PredictorError(Exception):
    """Base class for exceptions in this module."""

    pass


class BasePredictor:
    """Predictors forcast the next point in the time series of a player's points."""

    def __init__(
        self, name: str = "Unnamed Predictor", needs_training: bool = False
    ) -> None:
        """Initialise the predictor.

        Args:
            name (str, optional): Name for the predictor. Defaults to "Unnamed Predictor".
            needs_training (bool, optional): Whether or not the model needs training or not. Defaults to False.
        """
        self.name = name
        self.needs_training = needs_training

    def train(self, x: npt.NDArray) -> None:
        """Train the predictor - should be used on all the data."""
        pass

    @abstractmethod
    def predict(self, x: npt.NDArray) -> npt.ArrayLike:
        """Predict the next points in all the time series.

        Args:
            x (npt.NDArray): Array of time series

        Returns:
            npt.ArrayLike: Array of the next point for each row in the time series
        """
        pass


class LinearPredictor(BasePredictor):
    """A predictor that uses linear regression to predict the next point in the time series."""

    def __init__(self) -> None:
        """Initialise the predictor."""
        super().__init__("LinearRegression")

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Predict the next points in all the time series.

        Args:
            x (npt.NDArray): A nxm array of time series

        Returns:
            npt.NDArray: A nx1 array of the next point for each row in the time series
        """
        forcasts = []
        for x_row in x:
            self.model = LinearRegression()
            self.model.fit(np.array(range(len(x_row))).reshape(-1, 1), x_row)
            forcasts.append(self.model.predict(np.array(len(x_row)).reshape(-1, 1)))

        return forcasts


class ARIMAPredictor(BasePredictor):
    """Predictor that uses the ARIMA model from statsmodels."""

    def __init__(self, p: int = 5, d: int = 1, q: int = 0) -> None:
        """Initialise the predictor.

        Args:
            p (int, optional): The number of lag observations included in the model, also called the lag order. Defaults to 5.
            d (int, optional): The number of times that the raw observations are differenced, also called the degree of differencing. Defaults to 1.
            q (int, optional): The size of the moving average window, also called the order of moving average. Defaults to 0.
        """
        super().__init__("ARIMA")
        self.p = p
        self.d = d
        self.q = q

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Predict the next points in all the time series.

        Args:
            x (npt.NDArray): A nxm array of time series

        Raises:
            E: If the model fails to fit

        Returns:
            npt.NDArray: nx1 array of the next point for each row in the time series
        """
        forcasts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x_row in x:
                self.model = ARIMA(x_row, order=(self.p, self.d, self.q))
                self.model.initialize_approximate_diffuse()
                try:
                    self.fitted_model = self.model.fit()
                except Exception as E:
                    print(x_row)
                    raise E

                forcasts.append(self.fitted_model.forecast()[0])
        return forcasts


class LSTMPredictor(BasePredictor):
    """Predictor using an LSTM model."""

    def __init__(self, no_seasons: int = 5):
        """Initialise the predictor.

        Args:
            no_seasons (int, optional): Number of seasons to use an an input to the LSTM. Defaults to 5.
        """
        super().__init__(name="LSTM", needs_training=True)

        self.no_seasons = no_seasons
        self.model = keras.Sequential(
            [
                keras.layers.LSTM(64, input_shape=(self.no_seasons, 1)),
                keras.layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, x: npt.NDArray) -> None:
        """Train the LSTM model.

        Args:
            x (npt.NDArray): Array of time series
        """
        y = x[:, -1]
        x = x[:, -(1 + self.no_seasons) : -1]
        self.model.fit(x, y, epochs=100, verbose=0)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Predict using the LSTM model.

        Args:
            x (npt.NDArray): Array of time series

        Returns:
            npt.NDArray: Next point for each row in the time series
        """
        x = x[:, -self.no_seasons :]
        return self.model.predict(x).flatten()
