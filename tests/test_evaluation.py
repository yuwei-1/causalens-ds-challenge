import os
import shutil
import unittest
import polars as pl
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from src.evaluation import TimeSeriesCrossValidationExecutor


class TestTimeSeriesCrossValidationExecutor(unittest.TestCase):

    def test_run(self):
        
        num_features = 5
        feature_names = [f'feature_{i}' for i in range(num_features)]
        X, y = make_regression(n_samples=100, n_features=num_features, noise=0.1, random_state=42)
        df = pl.DataFrame(X, schema=feature_names).with_columns(
            pl.Series('target', y),
            pl.Series('group', [1] * X.shape[0])
        )

        exec = TimeSeriesCrossValidationExecutor(
            df,
            feature_names,
            'target',
            'group',
            KFold(5),
            model=XGBRegressor(),
            save_folder="./tests/data/fold_models"
        )

        exec.run()

        model_dir = os.listdir("./tests/data/fold_models")

        self.assertEqual(5, len(model_dir))


    def tearDown(self) -> None:
        if os.path.exists("./tests/data/fold_models"):
            shutil.rmtree("./tests/data/fold_models")