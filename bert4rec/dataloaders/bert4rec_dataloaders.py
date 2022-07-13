import abc
from absl import logging
import copy
import functools
import pandas as pd
import random
import string
import tensorflow as tf
import tensorflow_text as tf_text

from bert4rec.dataloaders import BaseDataloader
import bert4rec.dataloaders.dataloader_utils as utils
import bert4rec.tokenizers as tokenizers
import datasets.imdb as imdb
import datasets.ml_1m as ml_1m
import datasets.ml_20m as ml_20m
import datasets.reddit as reddit


class BERT4RecDataloader(BaseDataloader, abc.ABC):
    def __init__(self, max_predictions_per_batch: int = 5, max_seq_length: int = 128):
        self.lookup = None
        # BERT4Rec works with simple tokenizer
        self.tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        self._PAD_TOKEN = "[PAD]"
        self._START_TOKEN = "[CLS]"
        self._END_TOKEN = "[SEP]"
        self._MASK_TOKEN = "[MASK]"
        self._RANDOM_TOKEN = "[RANDOM]"
        self._UNK_TOKEN = "[UNK]"
        self._PAD_TOKEN_ID = self.tokenizer.tokenize(self._PAD_TOKEN)
        self._START_TOKEN_ID = self.tokenizer.tokenize(self._START_TOKEN)
        self._END_TOKEN_ID = self.tokenizer.tokenize(self._END_TOKEN)
        self._MASK_TOKEN_ID = self.tokenizer.tokenize(self._MASK_TOKEN)
        self._RANDOM_TOKEN_ID = self.tokenizer.tokenize(self._RANDOM_TOKEN)
        self._UNK_TOKEN_ID = self.tokenizer.tokenize(self._UNK_TOKEN)
        self._SPECIAL_TOKENS = [self._PAD_TOKEN, self._UNK_TOKEN, self._MASK_TOKEN, self._RANDOM_TOKEN,
                                self._START_TOKEN, self._END_TOKEN]
        self._MAX_PREDICTIONS_PER_BATCH = max_predictions_per_batch
        self._MAX_SEQ_LENGTH = max_seq_length

    def feature_preprocessing(self,
                              x: tf.int64,
                              sequence: list[str],
                              apply_mlm: bool = True,
                              finetuning: bool = False) -> dict:
        """
        Preprocess given features for training tasks (either regular training or finetuning)

        :param x: In this case depicts the user
        :param sequence: Depicts the sequence of items the user `x` interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing or not
        :return: A dictionary with the model inputs as well as a single additional entry for the original "labels"
        (this is the state of the tokenized input sequence before applying the masking language model)
        """
        # initiate return dictionary
        processed_features = dict()

        # tokenize segments to shape [num_items] each
        segments = [self.tokenizer.tokenize(sequence)]

        # truncate inputs to a maximum length
        _, trimmed_segments = utils.trim_content(self._MAX_SEQ_LENGTH, segments)

        # combine segments, get segment ids and add special tokens (i.e. start and end tokens)
        segments_combined, segment_ids = tf_text.combine_segments(
            trimmed_segments,
            start_of_sequence_id=self._START_TOKEN_ID,
            end_of_segment_id=self._END_TOKEN_ID
        )

        labels = copy.copy(segments_combined)

        # prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(
            segments_combined,
            max_seq_length=self._MAX_SEQ_LENGTH
        )
        input_type_ids, _ = tf_text.pad_model_inputs(
            segments_combined,
            max_seq_length=self._MAX_SEQ_LENGTH
        )

        # apply dynamic masking task
        if apply_mlm:
            if not finetuning:
                _, _, masked_token_ids, masked_lm_positions, masked_lm_ids = utils.apply_dynamic_masking_task(
                    segments_combined,
                    self._MAX_PREDICTIONS_PER_BATCH,
                    self._MASK_TOKEN_ID,
                    [self._START_TOKEN_ID, self._END_TOKEN_ID, self._UNK_TOKEN_ID, self._PAD_TOKEN_ID],
                    self.tokenizer.get_vocab_size())
            else:
                # only mask the last token (before the [SEP] token)
                tensor_values = segments_combined.numpy()[-1]
                masked_lm_ids = tf.ragged.constant([[tensor_values[-2]]], dtype=tf.int64)
                tensor_values[-2] = self._MASK_TOKEN_ID
                masked_token_ids = tf.ragged.constant([tensor_values], dtype=tf.int64)
                masked_lm_positions = tf.ragged.constant([[(len(tensor_values) - 2)]], dtype=tf.int64)

            # prepare and pad masked and combined segments
            input_word_ids, input_mask = tf_text.pad_model_inputs(
                masked_token_ids,
                max_seq_length=self._MAX_SEQ_LENGTH
            )
            input_type_ids, _ = tf_text.pad_model_inputs(
                masked_token_ids,
                max_seq_length=self._MAX_SEQ_LENGTH
            )

            # prepare and pad masking task inputs
            masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
                masked_lm_positions,
                max_seq_length=self._MAX_PREDICTIONS_PER_BATCH
            )
            masked_lm_ids, _ = tf_text.pad_model_inputs(
                masked_lm_ids,
                max_seq_length=self._MAX_PREDICTIONS_PER_BATCH
            )

            processed_features["masked_lm_ids"] = tf.squeeze(masked_lm_ids)
            processed_features["masked_lm_positions"] = tf.squeeze(masked_lm_positions)
            processed_features["masked_lm_weights"] = tf.squeeze(masked_lm_weights)

        # pad labels
        labels, _ = tf_text.pad_model_inputs(
            labels,
            max_seq_length=self._MAX_SEQ_LENGTH
        )

        processed_features["labels"] = tf.squeeze(labels)
        processed_features["input_word_ids"] = tf.squeeze(input_word_ids)
        processed_features["input_mask"] = tf.squeeze(input_mask)
        processed_features["input_type_ids"] = tf.squeeze(input_type_ids)

        return processed_features

    def call_feature_preprocessing(self, uid, sequence, apply_mlm: bool = True, finetuning: bool = False) -> list:
        """
        This function simply calls the `feature_preprocessing()` method and returns its return values as a list.
        This intermediate step allows the execution of python code in a tensorflow environment.
        Otherwise, the tokenizer could not alternate the given tensors during the `dataset.map()` procedure.

        :param uid: User ID
        :param sequence: Sequence of items the user interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing or not
        :return: List of feature_preprocessing() return values (instead of dictionary)
        """
        model_inputs = self.feature_preprocessing(uid, sequence, apply_mlm, finetuning)

        processed_features_list = []
        processed_features_list.append(model_inputs["labels"])
        processed_features_list.append(model_inputs["input_word_ids"])
        processed_features_list.append(model_inputs["input_mask"])
        processed_features_list.append(model_inputs["input_type_ids"])

        if apply_mlm:
            processed_features_list.append(model_inputs["masked_lm_ids"])
            processed_features_list.append(model_inputs["masked_lm_positions"])
            processed_features_list.append(model_inputs["masked_lm_weights"])

        # detailed declaration to ensure correct positions
        return processed_features_list

    def ds_map_fn(self, apply_mlm: bool, finetuning: bool, uid, sequence) -> dict:
        """
        See `call_feature_preprocessing()` method, why this intermediate step is necessary.

        :param uid: User ID
        :param sequence: Sequence of items the user interacted with
        :param apply_mlm: Determines, whether to apply the masked language model or not
        :param finetuning: Determines, whether to apply finetuning preprocessing or not
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

    def get_tokenizer(self):
        return self.tokenizer


class BERT4RecML1MDataloader(BERT4RecDataloader):
    def load_data(self) -> tf.data.Dataset:
        df = ml_1m.load_ml_1m()
        df = df.groupby("uid")
        user_grouped_df = pd.DataFrame(columns=["uid", "movies_sequence"])
        for user, u_data in df:
            user_seq = pd.DataFrame({"uid": user, "movies_sequence": [u_data["movie_name"].to_list()]})
            user_grouped_df = pd.concat([user_grouped_df, user_seq], ignore_index=True)
        datatypes = ["int64", "list"]
        ds = utils.convert_df_to_ds(user_grouped_df, datatypes)
        return ds

    def preprocess_dataset(self, ds: tf.data.Dataset = None, apply_mlm: bool = True, finetuning: bool = False) \
            -> tf.data.Dataset:
        if ds is None:
            ds = self.load_data()

        prepared_ds = ds.map(functools.partial(self.ds_map_fn, apply_mlm, finetuning))

        return prepared_ds


class BERT4RecML20MDataloader(BERT4RecDataloader):
    def load_data(self) -> tf.data.Dataset:
        df = ml_20m.load_ml_20m()
        df = df.groupby("uid")
        user_grouped_df = pd.DataFrame(columns=["uid", "movies_sequence"])
        for user, u_data in df:
            user_seq = pd.DataFrame({"uid": user, "movies_sequence": [u_data["movie_name"].to_list()]})
            user_grouped_df = pd.concat([user_grouped_df, user_seq], ignore_index=True)
        datatypes = ["int64", "list"]
        ds = utils.convert_df_to_ds(user_grouped_df, datatypes)
        return ds

    def preprocess_dataset(self, ds: tf.data.Dataset = None, apply_mlm: bool = True, finetuning: bool = False) \
            -> tf.data.Dataset:
        if ds is None:
            ds = self.load_data()

        prepared_ds = ds.map(functools.partial(self.ds_map_fn, apply_mlm, finetuning))

        return prepared_ds


class BERT4RecIMDBDataloader(BERT4RecDataloader):
    def load_data(self, apply_mlm: bool = True, finetuning: bool = False) -> tf.data.Dataset:
        raise NotImplementedError("The IMDB dataset is not (yet) implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")

    def preprocess_dataset(self, ds: tf.data.Dataset = None) -> tf.data.Dataset:
        raise NotImplementedError("The IMDB dataset is not (yet) implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")


class BERT4RecRedditDataloader(BERT4RecDataloader):
    def load_data(self, apply_mlm: bool = True, finetuning: bool = False) -> tf.data.Dataset:
        # df = reddit.load_reddit()
        raise NotImplementedError("The Reddit dataset is not yet implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")

    def preprocess_dataset(self, ds: tf.data.Dataset = None) -> tf.data.Dataset:
        raise NotImplementedError("The Reddit dataset is not yet implemented to be utilised in conjunction "
                                  "with the BERT4Rec model.")


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    dataloader = BERT4RecML1MDataloader()
    ds = dataloader.load_data()
    prepared_ds = dataloader.preprocess_dataset(ds, True, True)
    for x in prepared_ds.take(1):
        print(x)

    exit()

    test_data = tf.ragged.constant([random.choice(string.ascii_letters) for _ in range(25)])
    logging.debug(test_data)
    model_input = dataloader.feature_preprocessing(None, test_data, True)
    logging.debug(model_input)
    tensor = model_input["input_word_ids"]
    tokenizer = dataloader.get_tokenizer()
    detokenized = tokenizer.detokenize(tensor, ["[PAD]"])
    print(detokenized)
