import tensorflow as tf
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
        # df = reddit.load_reddit()
        raise NotImplementedError("The Reddit dataset is not yet implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")

    def load_data_into_split_ds(self, duplication_factor: int = None) \
            -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        raise NotImplementedError("The Reddit dataset is not (yet) implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")

    def generate_vocab(self, source=None) -> True:
        raise NotImplementedError("The Reddit dataset is not yet implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")

    def create_item_list(self) -> list:
        pass
