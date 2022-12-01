from absl import logging
import pandas as pd
import random
import string
import tensorflow as tf
from typing import Union

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils as utils
from bert4rec import tokenizers
import datasets.ml_1m as ml_1m


class BERT4RecML1MDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 200,
                 max_predictions_per_seq: int = 40,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = "simple",
                 min_sequence_len: int = 5):

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
        return "ml_1m"

    def load_data_into_ds(self) -> tf.data.Dataset:
        df = ml_1m.load_ml_1m()
        df = df.sort_values(by="timestamp")
        df = df.groupby("uid")
        user_grouped_df = pd.DataFrame(columns=["uid", "movies_sequence"])
        for user, u_data in df:
            user_seq = pd.DataFrame({"uid": user, "movies_sequence": [u_data["movie_name"].to_list()]})
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

        df = ml_1m.load_ml_1m()
        df = df.sort_values(by="timestamp")

        return utils.split_df_into_three_ds(df, duplication_factor, "uid", "movie_name")

    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        if source is None:
            df = ml_1m.load_ml_1m()
            source = set(df["movie_name"])
        super().generate_vocab(source, progress_bar)

    def create_item_list(self) -> list:
        df = ml_1m.load_ml_1m()
        return df["movie_name"].to_list()


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    dataloader = BERT4RecML1MDataloader()
    dataloader.generate_vocab()
    tokenizer = dataloader.get_tokenizer()
    ds = dataloader.load_data_into_ds()
    prepared_ds = dataloader.preprocess_dataset(ds, True, False)
    test_data = [random.choice(string.ascii_letters) for _ in range(25)]
    # logging.debug(test_data)
    model_input = dataloader.feature_preprocessing(None, test_data, True)
    # logging.debug(model_input)
    tensor = model_input["input_word_ids"]
    detokenized = tokenizer.detokenize(tensor, [dataloader._PAD_TOKEN])
    # print(detokenized)
    batched_ds = utils.make_batches(prepared_ds, buffer_size=100)
    for b in batched_ds.take(1):
        print(b)
