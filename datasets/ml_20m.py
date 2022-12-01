"""
This file provides the MovieLens 20M Dataset referenced in the BERT4Rec paper.
(https://arxiv.org/abs/1904.06690)
See: https://grouplens.org/datasets/movielens/20m/
"""
from absl import logging
import pandas as pd
import tqdm

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_ml_20m() -> pd.DataFrame:
    tqdm.tqdm.pandas()

    url = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    download_dir = utils.get_virtual_env_path().joinpath("data", "ml-20m")
    # size in bytes of the fully downloaded dataset
    download_size = 875588784

    if not dataset_utils.is_available(download_dir, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download_and_unpack_to_folder(url, download_dir, "zip", True)
    logging.info("Raw data already exists. Skip downloading")

    ratings_file_path = download_dir.joinpath("ratings.csv")
    df = pd.read_csv(ratings_file_path, header=0)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    movies_file_path = download_dir.joinpath('movies.csv')
    movies_df = pd.read_csv(movies_file_path, header=0)
    movies_df.columns = ['sid', 'movie_name', 'categories']
    logging.info("Merge dataframes")
    df = pd.merge(df, movies_df).progress_apply(lambda x: x)
    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = load_ml_20m()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
