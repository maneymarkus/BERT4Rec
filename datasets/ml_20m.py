"""
This class provides the MovieLens 1M Dataset.
See: https://grouplens.org/datasets/movielens/1m/
"""
from absl import logging
import pandas as pd

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_ml_20m() -> pd.DataFrame:
    url = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    download_dir = utils.get_virtual_env_path().joinpath("data", "ml-20m")
    # size in bytes of the fully downloaded dataset
    download_size = 875588784

    if not dataset_utils.is_available(download_dir, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download_and_unpack_to_folder(url, download_dir, "zip", True)
    logging.info("Raw data already exists. Skip downloading")

    ratings_file_path = download_dir.joinpath("ratings.csv")
    df = pd.read_csv(ratings_file_path, header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    movies_file_path = download_dir.joinpath('movies.csv')
    movies_df = pd.read_csv(movies_file_path, header=None)
    movies_df.columns = ['sid', 'movie_name', 'categories']
    df = pd.merge(df, movies_df)
    return df


def preprocess_ml_20m(dataframe: pd.DataFrame, min_rating: int = 0, min_uc: int = 5, min_sc: int = 0) \
        -> (pd.DataFrame, dict, dict):
    """
    Args:
        dataframe: The generated dataframe based on the ml_1m dataset
        min_rating: Determines the lowest rating that should be included in the dataset (rating from 1 to 5)
        min_uc: Determines how many ratings a user should have submitted in order to be included
        in the final dataset (users with less ratings than min_uc will be filtered out)
        min_sc: Determines how many ratings an item (motive) should have in order to be included
        in the final dataset (items with less ratings than min_sc will be filtered out)
    """
    logging.debug("Make implicit:\n%s\n", dataframe)
    df = make_implicit(dataframe, min_rating)
    logging.debug("Filter Users and Items:\n%s\n", df)
    df = filter_users_and_items(df, min_uc, min_sc)
    # Sort dataset by user and timestamp
    df = df.sort_values(["uid", "timestamp"])
    return df


def make_implicit(df: pd.DataFrame, min_rating: int = 0) -> pd.DataFrame:
    """
    This function only keeps rows in the dataframe with a rating higher than or equal to self.min_rating
    """
    logging.info('Turning into implicit ratings')
    df["rating"] = df["rating"].astype('int32')
    df = df[df['rating'] >= min_rating]
    return df


def filter_users_and_items(df: pd.DataFrame, min_uc: int = 5, min_sc: int = 0) \
        -> pd.DataFrame:
    """
    This function filters users and items.
    It groups users and items by id and removes users and items that appear less than
    self.min_sc for items and self.min_uc for users respectively.
    In other words: users that only rated less than self.min_uc items will be filtered out
    """
    logging.info('Filtering users and items')
    if min_uc > 0:
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        df = df[df['uid'].isin(good_users)]

    if min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        df = df[df['sid'].isin(good_items)]
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
