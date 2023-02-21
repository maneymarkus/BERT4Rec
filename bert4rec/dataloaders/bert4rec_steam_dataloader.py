from typing import Type, Union

from bert4rec.dataloaders import BERT4RecDataloader, preprocessors
from bert4rec import tokenizers
import datasets


class BERT4RecSteamDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 50,
                 max_predictions_per_seq: int = 20,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = "simple",
                 data_source: Type[datasets.BaseDataset] = datasets.Steam,
                 preprocessor: Type[preprocessors.BasePreprocessor] = preprocessors.BERT4RecPreprocessor,
                 masked_lm_prob: float = 0.4,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 min_sequence_len: int = 3):
        super().__init__(
            max_seq_len,
            max_predictions_per_seq,
            tokenizer,
            data_source,
            preprocessor,
            masked_lm_prob,
            mask_token_rate,
            random_token_rate,
            input_duplication_factor,
            min_sequence_len
        )

    @property
    def dataset_identifier(self):
        return "steam"

    def get_data(self,
                 split_data: bool = True,
                 sort_by: str = None,
                 extract_data: list = ["item_id"],
                 datatypes: list = ["list"],
                 duplication_factor: int = 5,
                 group_by: str = "user_id",
                 apply_mlm: bool = True,
                 finetuning_split: float = 0) -> tuple:
        return super().get_data(split_data,
                                sort_by,
                                extract_data,
                                datatypes,
                                duplication_factor,
                                group_by,
                                apply_mlm,
                                finetuning_split)

    def load_data(self,
                  split_data: bool = True,
                  sort_by: str = None,
                  extract_data: list = ["item_id"],
                  datatypes: list = ["list"],
                  duplication_factor: int = 5,
                  group_by: str = "user_id") -> tuple:
        return super().load_data(split_data,
                                 sort_by,
                                 extract_data,
                                 datatypes,
                                 duplication_factor,
                                 group_by)

    def prepare_training(self,
                         sort_by: str = None,
                         extract_data: list = ["item_id"],
                         datatypes: list = ["list"],
                         group_by: str = "user_id",
                         finetuning_split: float = 0.1) -> tuple:
        return super().prepare_training(sort_by, extract_data, datatypes, group_by, finetuning_split)

    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        if source is None:
            df = self.data_source.load_data()
            source = set(df["item_id"])
        super().generate_vocab(source, progress_bar)

    def create_item_list(self) -> list:
        df = self.data_source.load_data()
        return df["item_id"].to_list()
