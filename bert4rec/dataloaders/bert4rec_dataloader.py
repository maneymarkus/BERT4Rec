from absl import logging
import tensorflow as tf
from typing import Union, Type

from bert4rec.dataloaders import BaseDataloader
from bert4rec.dataloaders import preprocessors
from bert4rec import tokenizers
import bert4rec.dataloaders.dataloader_utils as utils
from bert4rec.datasets import BaseDataset


class BERT4RecDataloader(BaseDataloader):
    """
    This class is not abstract as it may be instantiated for e.g. feature preprocessing without a specific
    dataset
    """

    def __init__(self,
                 max_seq_len: int,
                 max_predictions_per_seq: int,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = "simple",
                 data_source: Type[BaseDataset] = None,
                 preprocessor: Type[preprocessors.BasePreprocessor] = preprocessors.BERT4RecPreprocessor,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 min_sequence_len: int = 5):
        tokenizer = tokenizers.get(tokenizer)
        super().__init__(tokenizer, data_source, preprocessor)

        if input_duplication_factor < 1:
            raise ValueError("An input_duplication_factor of less than 1 is not allowed!")

        self._PAD_TOKEN = "[PAD]"
        self._MASK_TOKEN = "[MASK]"
        self._UNK_TOKEN = "[UNK]"
        self._PAD_TOKEN_ID = self.tokenizer.tokenize(self._PAD_TOKEN)
        self._MASK_TOKEN_ID = self.tokenizer.tokenize(self._MASK_TOKEN)
        self._UNK_TOKEN_ID = self.tokenizer.tokenize(self._UNK_TOKEN)
        self._SPECIAL_TOKENS = [self._PAD_TOKEN, self._UNK_TOKEN, self._MASK_TOKEN]
        # needs to be ordered for the creation of the prediction mask in BERT4Rec models
        self._SPECIAL_TOKEN_IDS = [self._PAD_TOKEN_ID, self._MASK_TOKEN_ID, self._UNK_TOKEN_ID]
        self._MAX_PREDICTIONS_PER_SEQ = max_predictions_per_seq
        self._MAX_SEQ_LENGTH = max_seq_len
        self.masked_lm_prob = masked_lm_prob
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.input_duplication_factor = input_duplication_factor
        self.min_sequence_len = min_sequence_len

    @property
    def dataset_identifier(self):
        raise NotImplementedError("The dataset_identifier method hasn't been implemented.")

    def get_data(self,
                 split_data: bool = True,
                 sort_by: str = None,
                 extract_data: list = [],
                 datatypes: list = [],
                 duplication_factor: int = None,
                 group_by: str = None,
                 apply_mlm: bool = True,
                 finetuning_split: float = 0) -> tuple:
        """
        Loads data from a "raw" source into a Dataset and performs preprocessing operations on it

        :param split_data: Determines if the data should be split (into training, validation and
            testing data). Depending on the given value, the tuple returned will either
            contain a single dataset or three
        :param sort_by: If given, sorts the dataframe by this column prior to extracting anything
        :param extract_data: The names of the columns that should be extracted into to returned
            dataset(s)
        :param datatypes: A list of datatypes describing the pd.DataFrame column values.
            Relevant for converting the DataFrame columns into datasets. Datatypes may be tried to
            be inferred but especially for sequence data (lists) this argument is necessary
        :param duplication_factor: If given, determines how many times the (training) dataset
            should be duplicated
        :param group_by: The column name of the dataframe to group by for the sequence creation
        :param apply_mlm: Determines whether to apply the masked language model or not
        :param finetuning_split: Determines how much of the training data should have finetuning
            preprocessing applied. Finetuning is always applied to the validation and test data.
        """
        if finetuning_split < 0 or finetuning_split > 1:
            raise ValueError(f"The finetuning_split argument has to be a float between 0 and 1. "
                             f"Given: {finetuning_split}")

        datasets = self.load_data(split_data,
                                  sort_by,
                                  extract_data,
                                  datatypes,
                                  duplication_factor,
                                  group_by)

        processed_datasets = []
        for i, ds in enumerate(datasets):
            if i >= 1:
                processed_datasets.append(self.process_data(ds, apply_mlm, finetuning=True))
            else:
                if finetuning_split > 0:
                    train_split = 1 - finetuning_split
                    train_ds, finetuning_train_ds, _ = utils.split_dataset(
                        ds, train_split=train_split, val_split=finetuning_split, test_split=0.0
                    )
                    train_ds = self.process_data(train_ds, finetuning=False)
                    finetuning_train_ds = self.process_data(finetuning_train_ds, finetuning=True)

                    train_ds = train_ds.concatenate(finetuning_train_ds)
                else:
                    train_ds = self.process_data(ds, apply_mlm, finetuning=False)
                processed_datasets.append(train_ds)

        return tuple(processed_datasets)

    def load_data(self,
                  split_data: bool = True,
                  sort_by: str = None,
                  extract_data: list = [],
                  datatypes: list = [],
                  duplication_factor: int = None,
                  group_by: str = None) -> tuple:

        if len(extract_data) != len(datatypes):
            raise ValueError(f"The length of the extract_data list ({len(extract_data)}) has to "
                             f"be the same as the length of the datatypes list ({len(datatypes)}).")

        df = self.data_source.load_data()
        if sort_by is not None:
            df = df.sort_values(by=sort_by)
        dfs = tuple()
        if not split_data:
            sequence_df = utils.make_sequence_df(df, group_column_name=group_by, extract_sequences=extract_data)
            dfs += (sequence_df,)
        else:
            dfs += utils.split_sequence_df(df, group_by, extract_data, self.min_sequence_len)
        datasets = []
        for df in dfs:
            datasets.append(utils.convert_df_to_ds(df, datatypes))
        if duplication_factor is None:
            duplication_factor = self.input_duplication_factor
        datasets[0] = utils.duplicate_dataset(datasets[0], duplication_factor)
        return tuple(datasets)

    def process_data(self, ds: tf.data.Dataset,
                     apply_mlm: bool = True,
                     finetuning: bool = False) -> tf.data.Dataset:
        self.preprocessor.set_properties(tokenizer=self.tokenizer,
                                         max_seq_len=self._MAX_SEQ_LENGTH,
                                         max_predictions_per_seq=self._MAX_PREDICTIONS_PER_SEQ,
                                         mask_token_id=self._MASK_TOKEN_ID,
                                         unk_token_id=self._UNK_TOKEN_ID,
                                         pad_token_id=self._PAD_TOKEN_ID,
                                         masked_lm_rate=self.masked_lm_prob,
                                         mask_token_rate=self.mask_token_rate,
                                         random_token_rate=self.random_token_rate)
        prepared_ds = self.preprocessor.process_dataset(ds, apply_mlm, finetuning)
        return prepared_ds

    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        if source is None:
            raise ValueError(f"Need a source to get the vocab from!")

        logging.info("Start generating vocab")
        _ = self.tokenizer.tokenize(source, progress_bar)
        return True

    def prepare_training(self,
                         sort_by: str = None,
                         extract_data: list = [],
                         datatypes: list = [],
                         group_by: str = None,
                         finetuning_split: float = 0.1) -> tuple:
        if finetuning_split < 0 or finetuning_split > 1:
            raise ValueError("The finetuning_split argument has to be a float between 0 and 1. "
                             f"Given: {finetuning_split}")

        self.generate_vocab()

        return self.get_data(split_data=True,
                             sort_by=sort_by,
                             extract_data=extract_data,
                             datatypes=datatypes,
                             group_by=group_by,
                             finetuning_split=finetuning_split,
                             apply_mlm=True)

    def prepare_inference(self, data):
        """
        Prepares input for inference (adds a masked token to the end of the sequence to predict
        this token aka "the future" (this is the recommendation))

        :param data:
        :return:
        """
        self.preprocessor.set_properties(tokenizer=self.tokenizer,
                                         max_seq_len=self._MAX_SEQ_LENGTH,
                                         max_predictions_per_seq=self._MAX_PREDICTIONS_PER_SEQ,
                                         mask_token_id=self._MASK_TOKEN_ID,
                                         unk_token_id=self._UNK_TOKEN_ID,
                                         pad_token_id=self._PAD_TOKEN_ID,
                                         masked_lm_rate=self.masked_lm_prob,
                                         mask_token_rate=self.mask_token_rate,
                                         random_token_rate=self.random_token_rate)

        prepared_sequence = self.preprocessor.prepare_inference(data)

        return prepared_sequence

    def create_item_list(self) -> list:
        raise NotImplementedError("This method hasn't been implemented yet in this dataloader "
                                  "class.")
