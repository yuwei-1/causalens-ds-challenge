import datetime
from typing import Tuple
import polars as pl
from meteostat import Point, Hourly, Daily


class GetWeatherFeaturesForPeriod():

    def __init__(self,
                 start_dtime : datetime.datetime,
                 end_dtime : datetime.datetime,
                 interval : str = "1h",
                 time_col_name : str = "bucket",
                 location : Tuple[float] = (40.73, -73.933)) -> None:
        
        loc = Point(*location)
        if interval == "1h":
            data = Hourly(loc, start_dtime, end_dtime)
        elif interval == "1d":
            data = Daily(loc, start_dtime, end_dtime)
        else:
            raise NotImplementedError()
        
        data = data.fetch()

        data[time_col_name] = data.index
        self.weather_df = pl.from_pandas(data)

        self.time_col_name = time_col_name


    def add_weather_features(self, existing_data : pl.DataFrame):
        assert self.time_col_name in existing_data.columns, f"{self.time_col_name} not in existing_data columns"

        existing_time_unit = existing_data.schema[self.time_col_name].time_unit

        weather_features = self.weather_df.with_columns(
            pl.col(self.time_col_name).cast(pl.Datetime(time_unit=existing_time_unit))
        )

        return existing_data.join(weather_features, on="bucket")
    


def bike_station_optimisation_metrics(
        df : pl.DataFrame,
        id_col_name : str = "station_id",
        net_flow_col_name : str = "net_flow",
        time_col_name : str = "bucket"
    ):

    """
    For computing a proxy "utilisation metric" to score how
    well utilised a bike station is.

    The metric is the variance of occupancy, where occupancy is the fraction 
    if the station that is filled with bikes.

    The reason behind this is simply that under-utilised bike stations will
    tend to less users and thus have less variance in terms of occupancy.
    """

    df = df.sort(by=time_col_name, descending=False)
    df = df.with_columns(
        pl.col(net_flow_col_name).cum_sum().over(id_col_name).alias("relative_cum_net_flow")
    ).with_columns(
        pl.col("relative_cum_net_flow").max().over(id_col_name).alias("max_units"),
        pl.col("relative_cum_net_flow").min().over(id_col_name).alias("min_units")
    ).with_columns(
        (pl.col("max_units") - pl.col("min_units")).over(id_col_name).alias("estimated_station_size"),
        (pl.col("relative_cum_net_flow") - pl.col("min_units")).alias("estimated_bike_units_over_time")
    ).with_columns(
        (pl.col("estimated_bike_units_over_time") / pl.col("estimated_station_size")).over(id_col_name).alias("occupancy")
    ).with_columns(
        pl.col("estimated_bike_units_over_time").std().over(id_col_name).alias("utilisation_score"),
        pl.col("estimated_bike_units_over_time").last().over(id_col_name).alias("estimated_final_bike_units")
    )

    return df