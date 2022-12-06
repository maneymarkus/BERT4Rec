from absl import logging
import pandas as pd
import tensorflow as tf
from typing import Union

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils as utils
from bert4rec import tokenizers
import datasets.steam as steam


class BERT4RecSteamDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 50,
                 max_predictions_per_seq: int = 20,
                 masked_lm_prob: float = 0.4,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = "simple",
                 min_sequence_len: int = 3):

        super().__init__(
            max_seq_len,
            max_predictions_per_seq,
            masked_lm_prob,
            mask_token_rate,
            random_token_rate,
            input_duplication_factor,
            tokenizer,
            min_sequence_len
        )

    @property
    def dataset_identifier(self):
        return "steam"

    def load_data_into_ds(self) -> tf.data.Dataset:
        df = steam.load_steam()
        df = df.groupby("user_id")
        user_grouped_df = pd.DataFrame(columns=["user_id", "item_sequence"])
        for user, u_data in df:
            user_seq = pd.DataFrame({"user_id": user, "item_sequence": [u_data["item_id"].to_list()]})
            user_grouped_df = pd.concat([user_grouped_df, user_seq], ignore_index=True)
        datatypes = ["int64", "list"]
        ds = utils.convert_df_to_ds(user_grouped_df, datatypes)
        return ds

    def load_data_into_split_ds(self, duplication_factor: int = None) \
            -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Loads the represented dataset into three separate tf.data.Dataset objects (for training, validation
        and testing).

        :param duplication_factor: Determines how many times the training data set should be repeated
        to generate more samples
        :return:
        """
        super().load_data_into_split_ds(duplication_factor)
        if duplication_factor is None:
            duplication_factor = self.input_duplication_factor

        df = steam.load_steam()

        return utils.split_df_into_three_ds(df, duplication_factor, "user_id", "item_id")

    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        if source is None:
            df = steam.load_steam()
            source = set(df["item_id"])
        super().generate_vocab(source, progress_bar)

    def create_item_list(self) -> list:
        df = steam.load_steam()
        return df["item_id"].to_list()
