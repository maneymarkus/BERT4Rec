"""
This class provides the MovieLens 1M Dataset.
See: https://grouplens.org/datasets/movielens/1m/
"""
from absl import logging
import numpy as np
import pandas as pd
import tqdm

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_ml_1m() -> pd.DataFrame:
    tqdm.tqdm.pandas()

    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    download_dir = utils.get_virtual_env_path().joinpath("data", "ml-1m")
    # size in bytes of the fully downloaded dataset
    download_size = 24905384

    if not dataset_utils.is_available(download_dir, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download_and_unpack_to_folder(url, download_dir, "zip", True)
    logging.info("Raw data already exists. Skip downloading")

    ratings_file_path = download_dir.joinpath("ratings.dat")
    df = pd.read_csv(ratings_file_path, sep='::', header=None, engine="python", encoding="iso-8859-1")
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    movies_file_path = download_dir.joinpath('movies.dat')
    movies_df = pd.read_csv(movies_file_path, sep='::', header=None, engine="python", encoding="iso-8859-1")
    movies_df.columns = ['sid', 'movie_name', 'categories']
    logging.info("Merge dataframes")
    df = pd.merge(df, movies_df).progress_apply(lambda x: x)
    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = load_ml_1m()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
