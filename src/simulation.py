import math
import logging
from typing import *
import polars as pl
from copy import deepcopy
from pathlib import Path
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from src.feature_engineering import bike_station_optimisation_metrics
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class WalkForwardSimulationRunner:

    def __init__(self,
                 train_data : pl.DataFrame,
                 test_data : pl.DataFrame,
                 model : BaseEstimator,
                 training_col_names : List[str],
                 target : str,
                 group_col_name : str = "group",
                 net_flow_col_name : str = "net_flow",
                 walk_forward_interval : int = 7,
                 training_interval : int = 30,
                 plot_save_path : str = "./plots"
                 ) -> None:
        
        self.train_data = train_data
        self.test_data = test_data.sort(group_col_name)
        self.model = model
        self.training_col_names = training_col_names
        self.target = target
        self.group_col_name = group_col_name
        self.net_flow_col_name = net_flow_col_name
        self.walk_forward_interval = walk_forward_interval
        self.training_interval = training_interval
        self.plot_save_path = Path(plot_save_path)

    def _get_station_bike_quantities(self) -> Dict[float, float]:
        metrics = bike_station_optimisation_metrics(
            self.train_data
        )
        grouped = metrics.group_by("station_id").head(1)
        current_bikes_available = dict(zip(grouped['station_id'], grouped['estimated_final_bike_units']))
        return current_bikes_available
    
    def _non_ml_strategy(self, actual_bike_counts : Dict[float, int]) -> Dict:
        adjusted_bike_counts = deepcopy(actual_bike_counts)
        total_bikes = sum(actual_bike_counts.values())
        avg_bikes = total_bikes / len(actual_bike_counts)

        moved = 0

        not_assigned = total_bikes
        for i, key in enumerate(adjusted_bike_counts.keys()):
            if i == len(adjusted_bike_counts) - 1:
                adjusted_bike_counts[key] = not_assigned
                moved += max(adjusted_bike_counts[key] - actual_bike_counts[key], 0)
                break
            else:
                adjusted_bike_counts[key] = math.floor(avg_bikes)
                moved += max(adjusted_bike_counts[key] - actual_bike_counts[key], 0)
                not_assigned -= math.floor(avg_bikes)


        print(f"Number of bikes that need to be moved (average method): {moved}")

        assert sum(adjusted_bike_counts.values()) == total_bikes, "Bike count mismatch!"
        return adjusted_bike_counts, moved
    
    def _naive_bike_count_shuffle(self, projected_lowest_stock : Dict[float, float], actual_bike_counts : Dict[float, int]) -> Dict:
        adjusted_bike_counts = deepcopy(actual_bike_counts)
        total_bikes = sum(actual_bike_counts.values())
        required_bikes = {k : math.ceil(-v) for k,v in projected_lowest_stock.items() if v < 0}
        surplus_bike_stations = {k : v for k,v in projected_lowest_stock.items() if v > 0}

        extra = sum(surplus_bike_stations.values())
        needed = sum(required_bikes.values())
        rough_prop_needed = needed / extra

        print(f"Number of bikes that need to be moved (ML): {needed}")

        for key in required_bikes.keys():
            adjusted_bike_counts[key] += required_bikes[key]

        for key in surplus_bike_stations.keys():
            take_away = min(math.ceil(surplus_bike_stations[key] * rough_prop_needed), needed)
            adjusted_bike_counts[key] -= take_away
            needed -= take_away
            if needed == 0:
                break

        assert needed <= 0, "Not enough bikes to satisfy demand!"
        assert sum(adjusted_bike_counts.values()) == total_bikes, "Bike count mismatch!"
        return adjusted_bike_counts, sum(required_bikes.values())
    
    def _compute_supply_issues(self, actual_demand_per_station : Dict[float, float], bikes_per_station : Dict[float, float]) -> Tuple[int, int]:
        stockouts = 0
        for key in actual_demand_per_station.keys():
            bikes_present = bikes_per_station[key]
            worst_demand = actual_demand_per_station[key]
            if bikes_present + worst_demand < 0:
                stockouts += 1
        return stockouts
        
    def run(self):
        
        actual_quantities = self._get_station_bike_quantities()

        self.train_data = self.train_data.with_columns(
            pl.lit(0.0, dtype=pl.Float32).alias("predicted")
        )

        test_unique_groups = self.test_data[self.group_col_name].unique().to_list()
        train_unique_groups = self.train_data[self.group_col_name].unique().to_list()
        train_groups_used = train_unique_groups[-self.training_interval:]

        num_groups = len(test_unique_groups)
        steps = round(num_groups / self.walk_forward_interval)

        for step in range(steps):

            logger.info(f"Simulating one step into the future.")

            train_data = self.train_data.filter(pl.col(self.group_col_name).is_in(train_groups_used))
            X_train, y_train = train_data.select(self.training_col_names).to_pandas(), train_data[self.target].to_numpy()
            self.model.fit(X_train, y_train, verbose=False)

            test_window = test_unique_groups[:self.walk_forward_interval]
            test_unique_groups = test_unique_groups[self.walk_forward_interval:]
            test_group = self.test_data.filter(pl.col(self.group_col_name).is_in(test_window))

            print(f"Train interval is: {train_data[self.group_col_name].min()} to {train_data[self.group_col_name].max()}")
            print(f"Test interval is: {test_group[self.group_col_name].min()} to {test_group[self.group_col_name].max()}")

            predicted_net_flow = self.model.predict(test_group.select(self.training_col_names).to_pandas())
            test_group = test_group.with_columns(
                pl.Series("predicted", predicted_net_flow)
            )

            minimum_bike_est = test_group.with_columns(
                pl.col("predicted").cum_sum().over("station_id").alias("cum_predicted_net_flow"),
                pl.col(self.net_flow_col_name).cum_sum().over("station_id").alias("cum_actual_net_flow")
            )

            min_stock = minimum_bike_est.group_by("station_id").agg(
                pl.col("cum_predicted_net_flow").min().alias("min_predicted_stock"),
                pl.col("cum_actual_net_flow").min().alias("min_actual_stock"),
                pl.col("cum_actual_net_flow").last().alias("net_stock_change")
            )

            actual_demand_per_station = dict(zip(min_stock['station_id'], min_stock['min_actual_stock']))
            predicted_demand_per_station = dict(zip(min_stock['station_id'], min_stock['min_predicted_stock']))
            net_stock_change_per_station = dict(zip(min_stock['station_id'], min_stock['net_stock_change']))

            predicted_lowest_stock_point = {k: v + predicted_demand_per_station.get(k, 0) for k, v in actual_quantities.items()}
            ml_adjusted_quatities, ml_moved = self._naive_bike_count_shuffle(predicted_lowest_stock_point, actual_quantities)
            non_ml_adjusted_quantities, avg_moved = self._non_ml_strategy(actual_quantities)

            stockouts_without_adjust = self._compute_supply_issues(actual_demand_per_station, actual_quantities)
            stockouts_with_adjust = self._compute_supply_issues(actual_demand_per_station, ml_adjusted_quatities)
            stockouts_with_mean_adjust = self._compute_supply_issues(actual_demand_per_station, non_ml_adjusted_quantities)

            logger.info(f"Stockouts without adjustment: {stockouts_without_adjust}, with ML adjustment: {stockouts_with_adjust}")
            print(f"Stockouts without adjustment: {stockouts_without_adjust}, with ML adjustment: {stockouts_with_adjust}, with mean adjustment: {stockouts_with_mean_adjust}")
            print(f"Stockout reduction / bikes moved ratio (ML): {(stockouts_without_adjust - stockouts_with_adjust) / ml_moved}, (Mean): {(stockouts_without_adjust - stockouts_with_mean_adjust) / avg_moved}")

            actual_quantities = {k: v + net_stock_change_per_station.get(k) for k, v in actual_quantities.items()}


            sampled_stations = random.sample(list(min_stock['station_id']), 5)
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

            for ax, station_id in zip(axes, sampled_stations):
                station_data = minimum_bike_est.filter(pl.col('station_id') == station_id)
                
                ax.plot(station_data['cum_actual_net_flow'], label='Actual Net Flow', color='blue')
                ax.plot(station_data['cum_predicted_net_flow'], label='Predicted Net Flow', color='orange')
                ax.set_title(f'Station ID: {station_id}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Cumulative Net Flow')
                ax.legend()
                ax.grid()

            plt.tight_layout()
            plt.savefig(self.plot_save_path / f'step_{step}_sample_station_actual_vs_predicted_cumulative_net_flow.png')
            plt.close()
            
            
            plt.figure(figsize=(10, 6))
            plt.scatter(test_group[self.net_flow_col_name].to_numpy(), predicted_net_flow, label='Actual Net Flow', marker='o', alpha=0.5)
            plt.title('Predicted vs Actual Net Flow')
            plt.xlabel('Actual net flow')
            plt.ylabel('Predicted net flow')
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_save_path / f'step_{step}_predicted_vs_actual.png')
            plt.close()


            train_groups_used += test_window
            train_groups_used = train_groups_used[-self.training_interval:]
            self.train_data = pl.concat([self.train_data, test_group])
