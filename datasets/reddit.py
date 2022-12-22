"""
This file provides the Reddit Dataset.
See: https://files.pushshift.io/reddit/
"""
from absl import logging
import io
import json
import pandas as pd
import tqdm
import zstandard as zstd

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_reddit() -> pd.DataFrame:
    """
    Load reddit data from a bz2 compressed data file in json format

    :param debug:
    :return:
    """
    category = "comments"
    file_name = "RC_2011-01.zst"
    url = f"https://files.pushshift.io/reddit/{category}/{file_name}"
    download_file = utils.get_virtual_env_path().joinpath("data", "reddit", category,  file_name)
    # size in bytes of the fully downloaded dataset
    download_size = 621585706

    if not dataset_utils.is_available(download_file, download_size):
        logging.info("Raw data doesn't exist. Download...")
        dataset_utils.download(url, download_file)
    logging.info("Raw data already exists. Skip downloading")

    data = {}
    logging.info("Read data into python dictionary")
    with open(download_file, mode="rb") as json_file:
        # max window size to avoid memory overflow according to: https://stackoverflow.com/a/72092961
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(json_file)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for i, line in enumerate(tqdm.tqdm(text_stream)):
            data[i] = json.loads(line)
        json_file.close()

    logging.info("Start to convert dictionary data to pandas dataframe (this may take a while)")
    df = pd.DataFrame.from_dict(data, orient="index")

    df = filter_data(df)

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    # filter out [deleted] authors
    df = df[df["author"] != "[deleted]"]

    # filter out items that have less than three occurrences
    item_sequence_lengths = df.groupby("parent_id").size()
    filtered_items = item_sequence_lengths.index[item_sequence_lengths >= 3]
    df = df[df["parent_id"].isin(filtered_items)]

    # filter out users that have less than three occurrences
    user_sequence_lengths = df.groupby("author").size()
    filtered_users = user_sequence_lengths.index[user_sequence_lengths >= 3]
    df = df[df["author"].isin(filtered_users)]

    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = load_reddit()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
    print("\n\nVocabulary size (individual items):")
    print(len(set(data["parent_id"])))
    print("\nMaximum sequence length:")
    print(data.groupby("author").size().agg("max"))
    print("\nSequence length mean:")
    print(data.groupby("author").size().agg("mean"))
    print("\nSequence length median:")
    print(data.groupby("author").size().sort_values().agg("median"))
