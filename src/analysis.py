"""Functions for analysing the data."""
import typing as tp

import pandas as pd

from src.core import Season
from src.player import Player


def get_top_players_by_position(
    df: pd.DataFrame, position: str, n: int
) -> pd.DataFrame:
    """Get the top n players in the given position.

    Args:
        df (pd.DataFrame): Dataframe of all players
        position (str): Position to filter by
        n (int): Number of players

    Returns:
        pd.DataFrame: Top n players in the given position
    """
    return (
        df[df["element_type"] == position]
        .sort_values(by="total_points", ascending=False)
        .head(n)
    )


def get_cheapest_players_by_position(
    df: pd.DataFrame, position: str, n: int
) -> pd.DataFrame:
    """Get the cheapest n players in the given position.

    Args:
        df (pd.DataFrame): Dataframe of all players
        position (str): Position to filter by
        n (int): Number of players

    Returns:
        pd.DataFrame: Top n players in the given position
    """
    return (
        df[df["element_type"] == position]
        .sort_values(by="now_cost", ascending=True)
        .head(n)
    )


def get_player_points_by_year(
    df: pd.DataFrame, player_id: int, seasons: tp.List[Season]
) -> pd.DataFrame:
    """Get the points scored by the given player in each season.

    Args:
        df (pd.DataFrame): Dataframe of all players
        player_id (int): ID of the player

    Returns:
        pd.DataFrame: Points scored by the player in each season
    """
    player = Player.from_pandas_row(df[df["id"] == player_id].iloc[0])
    return player.get_historic_points_by_season(seasons)


def add_predicted_points_to_df(
    df: pd.DataFrame, seasons: tp.List[Season]
) -> pd.DataFrame:
    """Add a column to the dataframe with the predicted points for the next season.

    Args:
        df (pd.DataFrame): Dataframe of all players
        seasons (tp.List[Season]): Seasons to use for prediction

    Returns:
        pd.DataFrame: Dataframe with predicted points column
    """
    df["predicted_points"] = df.apply(
        lambda row: Player.from_pandas_row(row).predict_points_for_next_season(seasons),
        axis=1,
    )
    return df
