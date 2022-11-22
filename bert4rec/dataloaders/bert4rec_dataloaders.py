from absl import logging
import copy
import functools
import pandas as pd
import random
import string
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader
from bert4rec.tokenizers import BaseTokenizer
import bert4rec.dataloaders.dataloader_utils as utils
import bert4rec.tokenizers as tokenizers
import datasets.ml_1m as ml_1m
import datasets.ml_20m as ml_20m
import datasets.reddit as reddit


class BERT4RecDataloader(BaseDataloader):
    """
    This class is not abstract as it may be instantiated for e.g. feature preprocessing without a specific
    dataset
    """
    def __init__(self,
                 max_seq_len: int,
                 max_predictions_per_seq: int,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: BaseTokenizer = None):
        # BERT4Rec works with simple tokenizer
        if tokenizer is None:
            tokenizer = tokenizers.get("simple")
        super().__init__(tokenizer)

        if input_duplication_factor < 1:
            raise ValueError("An input_duplication_factor of less than 1 is not allowed!")

        self._PAD_TOKEN = "[PAD]"
        self._MASK_TOKEN = "[MASK]"
        self._RANDOM_TOKEN = "[RANDOM]"
        self._UNK_TOKEN = "[UNK]"
        self._PAD_TOKEN_ID = self.tokenizer.tokenize(self._PAD_TOKEN)
        self._MASK_TOKEN_ID = self.tokenizer.tokenize(self._MASK_TOKEN)
        self._RANDOM_TOKEN_ID = self.tokenizer.tokenize(self._RANDOM_TOKEN)
        self._UNK_TOKEN_ID = self.tokenizer.tokenize(self._UNK_TOKEN)
        self._SPECIAL_TOKENS = [self._PAD_TOKEN, self._UNK_TOKEN, self._MASK_TOKEN, self._RANDOM_TOKEN]
        # needs to be ordered for the creation of the prediction mask in BERT4Rec models
        self._SPECIAL_TOKEN_IDS = [self._PAD_TOKEN_ID, self._MASK_TOKEN_ID, self._RANDOM_TOKEN_ID, self._UNK_TOKEN_ID]
        self._MAX_PREDICTIONS_PER_SEQ = max_predictions_per_seq
        self._MAX_SEQ_LENGTH = max_seq_len
        self.masked_lm_prob = masked_lm_prob
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.input_duplication_factor = input_duplication_factor

    def dataset_identifier(self):
        pass

    def load_data_into_ds(self) -> tf.data.Dataset:
        pass

    def load_data_into_split_ds(self, duplication_factor: int = None) \
            -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        if duplication_factor is not None and duplication_factor < 1:
            raise ValueError(f"A duplication factor of less than 1 (given: {duplication_factor}) "
                             "is not allowed!")

    def generate_vocab(self, source=None) -> True:
        if source is None:
            raise ValueError(f"Need a source to get the vocab from!")

        _ = self.tokenizer.tokenize(source)
        return True

    def create_popular_item_ranking(self) -> list:
        pass

    def prepare_training(self, finetuning_split: float = 0.1):
        if finetuning_split < 0 or finetuning_split > 1:
            raise ValueError("The parameter finetuning_split can only be a float between 0 and 1 (including).")

        train_ds, val_ds, test_ds = self.load_data_into_split_ds(duplication_factor=self.input_duplication_factor)

        # make a small proportion of the dataset to have only the last item masked -> this works as finetuning
        if finetuning_split > 0:
            train_split = 1 - finetuning_split
            train_ds, finetuning_train_ds, _ = utils.split_dataset(
                train_ds, train_split=train_split, val_split=finetuning_split, test_split=0.0
            )
            train_ds = self.preprocess_dataset(train_ds)
            finetuning_train_ds = self.preprocess_dataset(finetuning_train_ds, finetuning=True)

            train_ds = train_ds.concatenate(finetuning_train_ds)
        else:
            train_ds = self.preprocess_dataset(train_ds)

        val_ds = self.preprocess_dataset(val_ds, finetuning=True)
        test_ds = self.preprocess_dataset(test_ds, finetuning=True)

        return train_ds, val_ds, test_ds

    def preprocess_dataset(self,
                           ds: tf.data.Dataset = None,
                           apply_mlm: bool = True,
                           finetuning: bool = False) -> tf.data.Dataset:
        """
        Preprocesses a given or the represented dataset

        :param ds: Given dataset to preprocess. Must have a similar format as the represented dataset or is the
            represented dataset
        :param apply_mlm: Whether to apply the masked language models preprocessing or not
        :param finetuning: Whether to create some samples (10% of the dataset) preprocessed for finetuning (only the
            last item is masked as specific training for the inference task later on)
        :return:
        """
        if ds is None:
            ds = self.load_data_into_ds()

        prepared_ds = ds.map(functools.partial(self.ds_map_fn, apply_mlm, finetuning))

        return prepared_ds

    def feature_preprocessing(self,
                              x,
                              sequence: list[str],
                              apply_mlm: bool = True,
                              finetuning: bool = False) -> dict:
        """
        Preprocess given features for training tasks (either regular training or finetuning)

        :param x: In this case depicts the user
        :param sequence: Depicts the sequence of items the user `x` interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing or not (mask only the last
            item in a sequence). Does not have an effect when apply_mlm param is set to false.
        :return: A dictionary with the model inputs as well as a single additional entry for the original "labels"
            (this is the state of the tokenized input sequence before applying the masking language model)
        """
        # initiate return dictionary
        processed_features = dict()

        segments = self.tokenizer.tokenize(sequence)

        # Truncate inputs to a maximum length. If finetuning is applied, take only the most recent items.
        # If sequence is actually shorter than the allowed length this expression will take the whole sequence
        if finetuning or len(segments) <= self._MAX_SEQ_LENGTH:
            segments = segments[-self._MAX_SEQ_LENGTH:]
        else:
            # If no finetuning is applied and the sequence is longer than _MAX_SEQ_LENGTH then take
            # a sequence of length _MAX_SEQ_LENGTH starting from random index
            start_i = random.randint(0, len(segments) - self._MAX_SEQ_LENGTH)
            segments = segments[start_i:start_i + self._MAX_SEQ_LENGTH]

        input_word_ids = tf.constant(segments, dtype=tf.int64)
        # build input mask
        input_mask = tf.ones_like(segments, dtype=tf.int64)
        input_type_ids = tf.zeros_like(segments, dtype=tf.int64)

        labels = copy.copy(input_word_ids)

        # pad inputs
        if self._MAX_SEQ_LENGTH - input_word_ids.shape[0] > 0:
            paddings = tf.constant([[self._MAX_SEQ_LENGTH - input_word_ids.shape[0], 0]])
            input_word_ids = tf.pad(input_word_ids, paddings)
            input_mask = tf.pad(input_mask, paddings)
            input_type_ids = tf.pad(input_type_ids, paddings)
            labels = tf.pad(labels, paddings)

        # apply dynamic masking task
        if apply_mlm:
            if not finetuning:
                # prepared segments is the masked_token_ids tensor (so with the mlm applied)
                input_word_ids, masked_lm_positions, masked_lm_ids = utils.apply_dynamic_masking_task(
                    input_word_ids,
                    self._MAX_PREDICTIONS_PER_SEQ,
                    self._MASK_TOKEN_ID,
                    [self._UNK_TOKEN_ID, self._PAD_TOKEN_ID],
                    self.tokenizer.get_vocab_size(),
                    selection_rate=self.masked_lm_prob,
                    mask_token_rate=self.mask_token_rate,
                    random_token_rate=self.random_token_rate)
            else:
                # only mask the last token
                tensor_values = input_word_ids.numpy()
                masked_lm_ids = tf.constant([tensor_values[-1]], dtype=tf.int64)
                tensor_values[-1] = self._MASK_TOKEN_ID
                input_word_ids = tf.constant(tensor_values, dtype=tf.int64)
                masked_lm_positions = tf.constant([(len(tensor_values) - 1)], dtype=tf.int64)

            masked_lm_weights = tf.ones_like(masked_lm_ids)

            # pad masked_lm inputs
            if masked_lm_ids.shape[0] < self._MAX_PREDICTIONS_PER_SEQ:
                paddings = tf.constant([[self._MAX_PREDICTIONS_PER_SEQ - masked_lm_ids.shape[0], 0]])
                masked_lm_ids = tf.pad(masked_lm_ids, paddings)
                masked_lm_positions = tf.pad(masked_lm_positions, paddings)
                masked_lm_weights = tf.pad(masked_lm_weights, paddings)

            processed_features["masked_lm_ids"] = masked_lm_ids
            processed_features["masked_lm_positions"] = masked_lm_positions
            processed_features["masked_lm_weights"] = masked_lm_weights

        processed_features["user_id"] = x
        processed_features["labels"] = labels
        processed_features["input_word_ids"] = input_word_ids
        processed_features["input_mask"] = input_mask
        processed_features["input_type_ids"] = input_type_ids

        return processed_features

    def call_feature_preprocessing(self, uid, sequence, apply_mlm: bool = True, finetuning: bool = False) -> list:
        """
        This function simply calls the `feature_preprocessing()` method and returns its return values as a list.
        This intermediate step allows the execution of python code in a tensorflow environment.
        Otherwise, the tokenizer could not alternate the given tensors during the `dataset.map()` procedure.

        :param uid: User ID
        :param sequence: Sequence of items the user interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing to 10% of the entries or not
        :return: List of feature_preprocessing() return values (instead of dictionary)
        """
        model_inputs = self.feature_preprocessing(uid, sequence, apply_mlm, finetuning)

        # detailed declaration to ensure correct positions
        processed_features_list = []
        processed_features_list.append(model_inputs["labels"])
        processed_features_list.append(model_inputs["input_word_ids"])
        processed_features_list.append(model_inputs["input_mask"])
        processed_features_list.append(model_inputs["input_type_ids"])

        if apply_mlm:
            processed_features_list.append(model_inputs["masked_lm_ids"])
            processed_features_list.append(model_inputs["masked_lm_positions"])
            processed_features_list.append(model_inputs["masked_lm_weights"])

        return processed_features_list

    def ds_map_fn(self, apply_mlm: bool, finetuning: bool, uid, sequence) -> dict:
        """
        See `call_feature_preprocessing()` method, why this intermediate step is necessary.

        :param uid: User ID
        :param sequence: Sequence of items the user interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing to 10% of the entries or not
        :return: Dictionary with the expected input values for the Transformer encoder
        """
        processed_features = dict()

        if apply_mlm:
            output = tf.py_function(
                self.call_feature_preprocessing,
                [uid, sequence, True, finetuning],
                [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64]
            )
            processed_features["masked_lm_ids"] = output[4]
            processed_features["masked_lm_positions"] = output[5]
            processed_features["masked_lm_weights"] = output[6]
        else:
            output = tf.py_function(
                self.call_feature_preprocessing,
                [uid, sequence, False, finetuning],
                [tf.int64, tf.int64, tf.int64, tf.int64]
            )

        processed_features["labels"] = output[0]
        processed_features["input_word_ids"] = output[1]
        processed_features["input_mask"] = output[2]
        processed_features["input_type_ids"] = output[3]

        return processed_features

    def prepare_inference(self, sequence: list[str]):
        """
        Prepares input (a sequence = a list of strings = the history) for inference (adds a masked token
        to the end of the sequence to predict this token)

        :param sequence:
        :return:
        """

        # if sequence is longer than max_seq_length - 1 (be able to append mask token) trim it
        # HINT: remove elements from beginning since we want to have the most recent history to "predict the future"
        sequence = sequence[-self._MAX_SEQ_LENGTH + 1:]

        # add [MASK] token to end of sequence
        sequence.append(self._MASK_TOKEN)

        preprocessed_sequence = self.feature_preprocessing(None, sequence, False, False)
        # remove user id key from dict as it is None
        del preprocessed_sequence["user_id"]

        # expand dimension of tensors since encoder needs inputs of dimension 2
        for key, value in preprocessed_sequence.items():
            if tf.is_tensor(value):
                preprocessed_sequence[key] = tf.expand_dims(value, axis=0)

        # no masked lm applied here, as inference does not have a ground truth

        return preprocessed_sequence


class BERT4RecML1MDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 200,
                 max_predictions_per_seq: int = 40,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: BaseTokenizer = None):

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

        return utils.load_movielens_data_in_split_ds(df, duplication_factor)

    def generate_vocab(self, source=None) -> True:
        if source is None:
            df = ml_1m.load_ml_1m()
            source = set(df["movie_name"])
        super(BERT4RecML1MDataloader, self).generate_vocab(source)

    def create_popular_item_ranking(self) -> list:
        df = ml_1m.load_ml_1m()
        item_list = df["movie_name"].to_list()
        sorted_item_list = utils.rank_items_by_popularity(item_list)
        return sorted_item_list


class BERT4RecML20MDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 200,
                 max_predictions_per_seq: int = 40,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: BaseTokenizer = None):

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
        return "ml_20m"

    def load_data_into_ds(self) -> tf.data.Dataset:
        df = ml_20m.load_ml_20m()
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

        df = ml_20m.load_ml_20m()

        return utils.load_movielens_data_in_split_ds(df, duplication_factor)

    def generate_vocab(self, source=None) -> True:
        if source is None:
            df = ml_20m.load_ml_20m()
            source = set(df["movie_name"])
        super(BERT4RecML20MDataloader, self).generate_vocab(source)

    def create_popular_item_ranking(self) -> list:
        df = ml_20m.load_ml_20m()
        item_list = df["movie_name"].to_list()
        sorted_item_list = utils.rank_items_by_popularity(item_list)
        return sorted_item_list


class BERT4RecRedditDataloader(BERT4RecDataloader):
    def __init__(self,
                 max_seq_len: int = 200,
                 max_predictions_per_seq: int = 40,
                 masked_lm_prob: float = 0.2,
                 mask_token_rate: float = 1.0,
                 random_token_rate: float = 0.0,
                 input_duplication_factor: int = 1,
                 tokenizer: BaseTokenizer = None):
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

    def create_popular_item_ranking(self) -> list:
        pass


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
