"""
This class provides the MovieLens 1M Dataset referenced in the BERT4Rec paper.
(https://arxiv.org/abs/1904.06690)
See: https://grouplens.org/datasets/movielens/1m/
"""
from absl import logging
import pandas as pd
import tqdm

from bert4rec.datasets import BaseDataset, dataset_utils
import bert4rec.utils as utils


class ML1M(BaseDataset):

    source = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    dest = utils.get_virtual_env_path().joinpath("data", "ml-1m")

    @classmethod
    def load_data(cls) -> pd.DataFrame:
        tqdm.tqdm.pandas()
        return super().load_data()

    @classmethod
    def is_available(cls) -> bool:
        # size in bytes of the fully downloaded dataset
        download_size = 24905384

        if not dataset_utils.check_availability_via_download_size(cls.dest, download_size):
            return False
        return True

    @classmethod
    def download(cls):
        dataset_utils.download_and_unpack_to_folder(cls.source, cls.dest, "zip", True)

    @classmethod
    def extract_data(cls) -> pd.DataFrame:
        ratings_file_path = cls.dest.joinpath("ratings.dat")
        df = pd.read_csv(ratings_file_path,
                         sep='::',
                         header=None,
                         engine="python",
                         encoding="iso-8859-1",
                         nrows=cls.load_n_records)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        movies_file_path = cls.dest.joinpath('movies.dat')
        movies_df = pd.read_csv(movies_file_path,
                                sep='::',
                                header=None,
                                engine="python",
                                encoding="iso-8859-1",
                                nrows=cls.load_n_records)
        movies_df.columns = ['sid', 'movie_name', 'categories']
        logging.info("Merge dataframes")
        df = pd.merge(df, movies_df).progress_apply(lambda x: x)
        return df


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    data = ML1M.load_data()
    print("Data Overview:\n")
    print(data)
    print("\n\nAvailable columns:\n")
    print(data.columns)
    print("\n\nAn example row:\n")
    print(data.iloc[0, :])
