import polars as pl
import numpy as np
from typing import *
from pathlib import Path
from joblib import dump
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


class TimeSeriesCrossValidationExecutor:

    def __init__(self,
                 data : pl.DataFrame,
                 training_features : List[str],
                 target : str,
                 group_col_name : str,
                 cross_validator : BaseCrossValidator,
                 model : BaseEstimator,
                 save_folder : str = "./models",
                 metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [mean_absolute_error]
                 ) -> None:
        self.data = data
        self.training_features = training_features
        self.target = target
        self.group_col_name = group_col_name
        self.cross_validator = cross_validator
        self.model = model
        self.save_folder = Path(save_folder)
        self.metrics = metrics

        if not self.save_folder.exists():
            self.save_folder.mkdir()

    def run(self):
        
        X = self.data.select(self.training_features).to_pandas()
        y = self.data[self.target].to_numpy()
        groups = self.data[self.group_col_name].to_numpy()

        fold = 0
        metric_results = {metric.__name__: [] for metric in self.metrics}
        oof_preds = np.full(len(X), np.nan)

        for train_idx, val_idx in self.cross_validator.split(X, groups=groups):
            fold += 1
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

            preds = self.model.predict(X_val)
            dump(self.model, self.save_folder / f"model_fold_{fold}.joblib")

            oof_preds[val_idx] = preds
            
            print(f"Fold {fold} results:")
            for metric in self.metrics:
                score = metric(y_val, preds)
                metric_results[metric.__name__].append(score)
                print(f"{metric.__name__}: {score:.3f}", end=" ")
            print()

        for metric_name, scores in metric_results.items():
            print(f"Average {metric_name} across folds:", np.mean(scores))

        return self.model, oof_preds, metric_results