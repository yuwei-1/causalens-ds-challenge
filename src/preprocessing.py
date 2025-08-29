from typing import List
import polars as pl


class BasePreprocessor():

    def __init__(self,
                 data : pl.DataFrame,
                 start_time_col_name : str = "started_at",
                 end_time_col_name : str = "ended_at",
                 process_null_columns : List[str] = ['start_station_id', 'end_station_id']
                 ) -> None:

        data = data.with_columns(
            pl.col(start_time_col_name).str.strptime(pl.Datetime).alias(start_time_col_name),
            pl.col(end_time_col_name).str.strptime(pl.Datetime).alias(end_time_col_name),
        )
        self.data = data.drop_nulls(subset=process_null_columns)


class NetFlowBinner(BasePreprocessor):

    """
    Bins events into regular time interval for
    each station.
    """

    departure_col_name = "departures"
    arrival_col_name = "arrivals"
    net_flow_col_name = "net_flow"
    id_col_name = "station_id"
    time_bucket_col_name = "bucket"

    def __init__(self,
                 data : pl.DataFrame,
                 interval : str,
                 start_time_col_name : str = "started_at",
                 end_time_col_name : str = "ended_at",
                 start_location_id : str = "start_station_id",
                 end_location_id : str = "end_station_id",
                 ) -> None:
        
        super().__init__(data, start_time_col_name, end_time_col_name, [start_location_id, end_location_id])

        self.interval = interval
        self.start_time_col_name = start_time_col_name
        self.end_time_col_name = end_time_col_name
        self.start_location_id = start_location_id
        self.end_location_id = end_location_id

    def process(self):

        _s, _e = "bucket_start", "bucket_end"

        self.data = self.data.with_columns([
            pl.col(self.start_time_col_name).dt.truncate(self.interval).alias(_s),
            pl.col(self.end_time_col_name).dt.truncate(self.interval).alias(_e),
        ])

        last = self.data[[_s, _e]].max().max_horizontal().item()
        first = self.data[[_s, _e]].min().min_horizontal().item()

        buckets = pl.datetime_range(
            start=first,
            end=last,
            interval=self.interval,
            eager=True
        ).to_frame(self.time_bucket_col_name)

        arrivals = self.data.group_by([self.end_location_id, _e]).agg(pl.len().alias(self.arrival_col_name))
        departures = self.data.group_by([self.start_location_id, _s]).agg(pl.len().alias(self.departure_col_name))
        arrivals = arrivals.rename({self.end_location_id: self.id_col_name, _e: self.time_bucket_col_name})
        departures = departures.rename({self.start_location_id: self.id_col_name, _s: self.time_bucket_col_name})

        unique_ids = self.data.select([self.start_location_id, self.end_location_id]).unpivot().select("value").unique()
        all_station_ids = pl.Series(self.id_col_name, unique_ids).to_frame()
        full_station_time_buckets = buckets.join(all_station_ids, how="cross").sort(by=[self.id_col_name,self.time_bucket_col_name])

        full_station_time_buckets = full_station_time_buckets.join(arrivals, on=[self.id_col_name,self.time_bucket_col_name], how="left").fill_null(0)
        full_station_time_buckets = full_station_time_buckets.join(departures, on=[self.id_col_name,self.time_bucket_col_name], how="left").fill_null(0)
        full_station_time_buckets = full_station_time_buckets.with_columns(
            (pl.col(self.arrival_col_name) - pl.col(self.departure_col_name).cast(int)).alias(self.net_flow_col_name)
        )
        return full_station_time_buckets