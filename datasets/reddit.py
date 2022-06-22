"""
This class provides the MovieLens 1M Dataset.
See: https://grouplens.org/datasets/movielens/1m/
"""
from absl import logging
import bz2
import json
import pandas as pd

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_reddit(debug: bool = False) -> pd.DataFrame:
    url = 'https://files.pushshift.io/reddit/comments/RC_2011-01.bz2'
    download_file = utils.get_virtual_env_path().joinpath("data", "reddit", "comments",  "RC_2011-01.bz2")
    # size in bytes of the fully downloaded dataset
    download_size = 613090022

    if not dataset_utils.is_available(download_file, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download(url, download_file)
    logging.info("Raw data already exists. Skip downloading")

    processed_input = []
    with bz2.open(download_file, encoding="utf-8", mode="rt") as json_file:
        if debug:
            lines = json_file.readlines(1000000)
        else:
            lines = json_file.readlines()
        for line in lines:
            processed_input.append(json.loads(line))
        json_file.close()
    df = pd.json_normalize(processed_input)
    return df
