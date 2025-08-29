import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def plot_cumulative_station_capacity(df : pl.DataFrame, station_name : str):

    """
    Plot the relative number of bikes at a station over time.
    """

    start_col_name = "start_station_name"
    end_col_name = "end_station_name"

    outbounds = (df[start_col_name] == station_name).cast(int)
    inbounds = (df[end_col_name] == station_name).cast(int)
    change = inbounds - outbounds
    number_of_bikes_at_station  = np.cumsum(change.fill_null(0).to_numpy())

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(number_of_bikes_at_station.shape[0]), number_of_bikes_at_station, label="Number of bikes at station")
    plt.show()