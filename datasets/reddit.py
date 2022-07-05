"""
This class provides the MovieLens 1M Dataset.
See: https://grouplens.org/datasets/movielens/1m/
"""
from absl import logging
import io
import json
import pandas as pd
import zstandard as zstd

import datasets.dataset_utils as dataset_utils
import bert4rec.utils as utils


def load_reddit(debug: bool = False) -> pd.DataFrame:
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

    processed_input = []
    with open(download_file, mode="rb") as json_file:
        # max window size to avoid memory overflow according to: https://stackoverflow.com/a/72092961
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(json_file)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        line_counter = 1
        for line in text_stream:
            if debug and line_counter > 100000:
                break
            processed_input.append(json.loads(line))
            line_counter += 1
        json_file.close()
    df = pd.json_normalize(processed_input)
    return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = load_reddit(True)
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
