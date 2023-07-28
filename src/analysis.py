import pandas as pd


def get_top_players_by_position(df: pd.DataFrame, position: str, n: int) -> pd.DataFrame:
    """Get the top n players in the given position."""
    return df[df['element_type'] == position].sort_values(by='total_points', ascending=False).head(n)

def get_cheapest_players_by_position(df: pd.DataFrame, position: str, n: int) -> pd.DataFrame:
    """Get the cheapest n players in the given position."""
    return df[df['element_type'] == position].sort_values(by='now_cost', ascending=True).head(n)
