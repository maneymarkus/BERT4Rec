"""
This file provides the Steam dataset referenced in the BERT4Rec paper
(https://arxiv.org/abs/1904.06690)
See: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
"""
from absl import logging
import gzip
import json
import pandas as pd
import tqdm

from bert4rec.datasets import BaseDataset, dataset_utils
import bert4rec.utils as utils


class Steam(BaseDataset):

    source = 'https://github.com/FeiSun/BERT4Rec/raw/master/data/steam.txt'
    dest = utils.get_virtual_env_path().joinpath("data", "steam", "ratings_steam_tokenized.txt")

    @classmethod
    def is_available(cls) -> bool:
        # size in bytes of the fully downloaded dataset
        download_size = 38226650

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


def load_steam_2() -> pd.DataFrame:
    """
    Tried to load the steam dataset from the official source. However, I don't know
    which one is the correct one and how the dataset has been processed.

    :return:
    """
    tqdm.tqdm.pandas()

    url = 'http://jmcauley.ucsd.edu/data/steam/australian_users_items.json.gz'
    download_dir = utils.get_virtual_env_path().joinpath("data", "steam")
    # size in bytes of the fully downloaded dataset

    download_size = 73574835

    if not dataset_utils.check_availability_via_download_size(download_dir, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download(url, download_dir.joinpath("ratings_steam.json.gz"))
    logging.info("Raw data already exists. Skip downloading")

    ratings_file_path = download_dir.joinpath("ratings_steam.json.gz")
    with gzip.open(ratings_file_path, "rb") as data_file:
        data = {}
        logging.info("Start parsing json data")
        for i, line in enumerate(tqdm.tqdm(data_file)):
            line = line.decode()
            line = line.replace("'", "\"")
            data[i] = json.loads(line)
    df = pd.DataFrame.from_dict(data, orient="index")
    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = Steam.load_data()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
