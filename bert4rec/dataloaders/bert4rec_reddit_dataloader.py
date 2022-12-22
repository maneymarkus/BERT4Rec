from absl import logging
import pandas as pd
import tensorflow as tf
import tqdm
from typing import Union

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils as utils
from bert4rec import tokenizers
import datasets.reddit as reddit


class BERT4RecRedditDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 200,
                 max_predictions_per_seq: int = 40,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = "simple"):

        super().__init__(
            max_seq_len,
            max_predictions_per_seq,
            masked_lm_prob,
            mask_token_rate,
            random_token_rate,
            input_duplication_factor,
            tokenizer)

    @property
    def dataset_identifier(self):
        return "reddit"

    def load_data_into_ds(self) -> tf.data.Dataset:
        df = reddit.load_reddit()
        df = df.sort_values(by="created_utc")
        df = df.groupby("author")
        data = dict()
        for user, u_data in tqdm.tqdm(df):
            user_seq = u_data["parent_id"].to_list()
            data[user] = user_seq
        datatypes = ["str", "list"]
        user_grouped_df = pd.DataFrame(list(data.items()), columns=["uid", "item_id"])
        ds = utils.convert_df_to_ds(user_grouped_df, datatypes)
        return ds

    def load_data_into_split_ds(self, duplication_factor: int = None) \
            -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):

        super().load_data_into_split_ds(duplication_factor)
        if duplication_factor is None:
            duplication_factor = self.input_duplication_factor

        df = reddit.load_reddit()
        df = df.sort_values(by="created_utc")
        datatypes = ["str", "list"]

        return utils.split_df_into_three_ds(df, duplication_factor, "author", "parent_id", datatypes)

    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        if source is None:
            df = reddit.load_reddit()
            source = set(df["parent_id"])
        super().generate_vocab(source, progress_bar)

    def create_item_list(self) -> list:
        df = reddit.load_reddit()
        return df["parent_id"].to_list()
