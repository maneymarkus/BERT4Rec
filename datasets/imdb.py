"""
This class provides the IMDB Movie Dataset
See: https://www.tensorflow.org/text/tutorials/classify_text_with_bert#download_the_imdb_dataset
"""

from absl import logging
import os
import pandas as pd
import pathlib
import tqdm
import typing

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


tqdm.tqdm.pandas()


def load_imdb() -> pd.DataFrame:
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    download_dir = utils.get_virtual_env_path().joinpath("data", "imdb")
    # size in bytes of the fully downloaded dataset
    download_size = 243767007

    if not dataset_utils.is_available(download_dir, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download_and_unpack_to_folder(url, download_dir, "tar", True)
    logging.info("Raw data already exists. Skip downloading")

    df_columns = ["text", "is_positive"]
    positive_test_dir = download_dir.joinpath("test/pos")
    negative_test_dir = download_dir.joinpath("test/neg")
    positive_train_dir = download_dir.joinpath("train/pos")
    negative_train_dir = download_dir.joinpath("train/neg")
    df = pd.DataFrame(columns=df_columns)

    logging.info("Read positive test data into dataframe.")
    df = iterate_directory(df, positive_test_dir, True)
    logging.info("Read negative test data into dataframe.")
    df = iterate_directory(df, negative_test_dir, False)
    logging.info("Read positive training data into dataframe.")
    df = iterate_directory(df, positive_train_dir, True)
    logging.info("Read negative training data into dataframe.")
    df = iterate_directory(df, negative_train_dir, False)
    print(df)
    return df


def iterate_directory(df: pd.DataFrame, dir_path: pathlib.Path, is_positive: bool) -> pd.DataFrame:
    for filename in tqdm.tqdm(os.listdir(dir_path)):
        file_path = dir_path.joinpath(filename)
        if os.path.isfile(file_path):
            content = file_get_content(file_path)
            new_row = {"text": content, "is_positive": 1 if is_positive else 0}
            df = df.append(new_row, ignore_index=True)
    return df


def file_get_content(file_path: pathlib.Path) -> typing.AnyStr:
    with open(file_path, "r", encoding="iso-8859-1") as file:
        return file.read()


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = load_imdb()
    print(data)
