import os
import polars as pl


def read_all_csvs(folder_path : str) -> pl.DataFrame:
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pl.read_csv(file_path, ignore_errors=True)
            dataframes.append(df)
    full_data = pl.concat(dataframes)
    return full_data
