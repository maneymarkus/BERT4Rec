"""
This file provides the Amazon Beauty dataset referenced in the BERT4Rec paper
(https://arxiv.org/abs/1904.06690)
See: https://nijianmo.github.io/amazon/index.html
"""
from absl import logging
import gzip
import json
import pandas as pd
import tqdm

from bert4rec.datasets import BaseDataset, dataset_utils
import bert4rec.utils as utils


class Beauty(BaseDataset):

    source = 'https://github.com/FeiSun/BERT4Rec/raw/master/data/beauty.txt'
    dest = utils.get_virtual_env_path().joinpath("data", "beauty", "ratings_beauty_tokenized.txt")

    @classmethod
    def is_available(cls) -> bool:
        # size in bytes of the fully downloaded dataset
        download_size = 3912093

        if not dataset_utils.check_availability_via_download_size(cls.dest, download_size):
            return False
        return True

    @classmethod
    def download(cls):
        dataset_utils.download(cls.source, cls.dest)

    @classmethod
    def extract_data(cls) -> pd.DataFrame:
        with open(cls.dest, "rb") as file:
            data = {}
            for i, line in enumerate(file.readlines()):
                if cls.load_n_records is not None and i >= cls.load_n_records:
                    break
                # first int -> user id; second int -> item id;
                parts = line.split()
                data[i] = {
                    # user_id can be saved as integer
                    "user_id": int(parts[0]),
                    # item_id has to be saved as str to use tokenizer
                    "item_id": parts[1],
                }

        df = pd.DataFrame.from_dict(data, orient="index")
        return df


def load_beauty_2(custom_filter: callable = None) -> pd.DataFrame:
    """
    Tried to load the beauty dataset from the official source. However, I don't know
    which one is the correct one and how the dataset has been processed.

    :param custom_filter:
    :return:
    """
    tqdm.tqdm.pandas()

    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz'
    download_dir = utils.get_virtual_env_path().joinpath("data", "beauty")
    ratings_file_path = download_dir.joinpath("ratings_beauty.json.gz")
    # size in bytes of the fully downloaded dataset
    download_size = 352748278

    if not dataset_utils.check_availability_via_download_size(ratings_file_path, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download(url, ratings_file_path)
    logging.info("Raw data already exists. Skip downloading")

    with gzip.open(ratings_file_path, "rb") as data_file:
        data = {}
        logging.info("Start parsing json data")
        for i, line in enumerate(tqdm.tqdm(data_file)):
            data[i] = json.loads(line)
    df = pd.DataFrame.from_dict(data, orient="index")

    if custom_filter is not None:
        df = custom_filter(df)

    return df


def load_beauty_3(custom_filter: callable = None) -> pd.DataFrame:
    """
    Just another try to load the beauty dataset from the official source.

    :param custom_filter:
    :return:
    """
    tqdm.tqdm.pandas()

    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv'
    download_dir = utils.get_virtual_env_path().joinpath("data", "beauty")
    ratings_file_path = download_dir.joinpath("ratings_beauty_2.csv")
    # size in bytes of the fully downloaded dataset
    download_size = 82432164

    if not dataset_utils.check_availability_via_download_size(ratings_file_path, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download(url, ratings_file_path)
    logging.info("Raw data already exists. Skip downloading")

    df = pd.read_csv(ratings_file_path, header=None, engine="python")
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    if custom_filter is not None:
        df = custom_filter(df)

    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = Beauty.load_data()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
