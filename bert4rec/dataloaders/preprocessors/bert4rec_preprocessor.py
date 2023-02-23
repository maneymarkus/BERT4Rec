import copy
import functools
import random
import tensorflow as tf

from .base_preprocessor import BasePreprocessor
from bert4rec import tokenizers
from bert4rec.dataloaders import dataloader_utils


class BERT4RecPreprocessor(BasePreprocessor):

    tokenizer: tokenizers.BaseTokenizer = None
    max_seq_len: int = None
    max_predictions_per_seq: int = None
    mask_token_id: int = None
    unk_token_id: int = None
    pad_token_id: int = None
    masked_lm_rate: float = None
    mask_token_rate: float = None
    random_token_rate: float = None

    @classmethod
    def set_properties(cls,
                       tokenizer: tokenizers.BaseTokenizer = None,
                       max_seq_len: int = None,
                       max_predictions_per_seq: int = None,
                       mask_token_id: int = None,
                       unk_token_id: int = None,
                       pad_token_id: int = None,
                       masked_lm_rate: float = None,
                       mask_token_rate: float = None,
                       random_token_rate: float = None):
        # the "if" and "or" statements make sure that setting only single values won't reset already
        # other set attributes
        cls.tokenizer = tokenizer or cls.tokenizer
        cls.max_seq_len = max_seq_len or cls.max_seq_len
        cls.max_predictions_per_seq = max_predictions_per_seq or cls.max_predictions_per_seq
        cls.mask_token_id = mask_token_id if mask_token_id is not None else cls.mask_token_id
        cls.unk_token_id = unk_token_id if unk_token_id is not None else cls.unk_token_id
        cls.pad_token_id = pad_token_id if pad_token_id is not None else cls.pad_token_id
        cls.masked_lm_rate = masked_lm_rate if masked_lm_rate is not None else cls.masked_lm_rate
        cls.mask_token_rate = mask_token_rate if mask_token_rate is not None else cls.mask_token_rate
        cls.random_token_rate = random_token_rate if random_token_rate is not None else cls.random_token_rate

    @classmethod
    def process_element(cls, sequence, apply_mlm: bool, finetuning: bool) -> dict:
        """
        Preprocess given features for training tasks (either regular training or finetuning)

        :param sequence: A single element of the dataset. In this case: a sequence (/list) of values
        """
        # initiate return dictionary
        processed_features = dict()

        tokens = cls.tokenizer.tokenize(sequence)

        # Truncate inputs to a maximum length. If finetuning is applied, take only the most recent items.
        # If sequence is actually shorter than the allowed length this expression will take the whole sequence
        if finetuning or len(tokens) <= cls.max_seq_len:
            segments = tokens[-cls.max_seq_len:]
        else:
            # If no finetuning is applied and the sequence is longer than _MAX_SEQ_LENGTH then take
            # a sequence of length _MAX_SEQ_LENGTH starting from random index
            start_i = random.randint(0, len(tokens) - cls.max_seq_len)
            segments = tokens[start_i:start_i + cls.max_seq_len]

        input_word_ids = tf.constant(segments, dtype=tf.int64)
        # build input mask
        input_mask = tf.ones_like(segments, dtype=tf.int64)

        labels = copy.copy(input_word_ids)
        # apply dynamic masking task
        if apply_mlm:
            if not finetuning:
                # prepared segments is the masked_token_ids tensor (so with the mlm applied)
                input_word_ids, masked_lm_positions, masked_lm_ids = dataloader_utils.apply_dynamic_masking_task(
                    input_word_ids,
                    cls.max_predictions_per_seq,
                    cls.mask_token_id,
                    [cls.unk_token_id, cls.pad_token_id],
                    cls.tokenizer.get_vocab_size(),
                    selection_rate=cls.masked_lm_rate,
                    mask_token_rate=cls.mask_token_rate,
                    random_token_rate=cls.random_token_rate)
            else:
                # only mask the last token
                input_word_ids, masked_lm_positions, masked_lm_ids = \
                    dataloader_utils.mask_last_token_only(input_word_ids, cls.mask_token_id)

            masked_lm_weights = tf.ones_like(masked_lm_ids)

            # pad masked_lm inputs
            if masked_lm_ids.shape[0] < cls.max_predictions_per_seq:
                paddings = tf.constant([[0, cls.max_predictions_per_seq - masked_lm_ids.shape[0]]])
                masked_lm_ids = tf.pad(masked_lm_ids, paddings, constant_values=cls.pad_token_id)
                masked_lm_positions = tf.pad(masked_lm_positions, paddings, constant_values=cls.pad_token_id)
                masked_lm_weights = tf.pad(masked_lm_weights, paddings, constant_values=cls.pad_token_id)

            processed_features["masked_lm_ids"] = masked_lm_ids
            processed_features["masked_lm_positions"] = masked_lm_positions
            processed_features["masked_lm_weights"] = masked_lm_weights

        # pad inputs
        if cls.max_seq_len - input_word_ids.shape[0] > 0:
            paddings = tf.constant([[0, cls.max_seq_len - input_word_ids.shape[0]]])
            input_word_ids = tf.pad(input_word_ids, paddings, constant_values=cls.pad_token_id)
            input_mask = tf.pad(input_mask, paddings, constant_values=cls.pad_token_id)
            labels = tf.pad(labels, paddings, constant_values=cls.pad_token_id)

        processed_features["labels"] = labels
        processed_features["input_word_ids"] = input_word_ids
        processed_features["input_mask"] = input_mask

        return processed_features

    @classmethod
    def process_dataset(cls, ds: tf.data.Dataset, apply_mlm: bool, finetuning: bool) -> tf.data.Dataset:
        return ds.map(functools.partial(
            cls._ds_map_fn, apply_mlm, finetuning
        ), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    @classmethod
    def prepare_inference(cls, data) -> dict:
        """
        Prepares input (a sequence = a list of strings = the history) for inference (adds a masked
        token to the end of the sequence to predict this token)

        :param data: The sequence that should be prepared. Should simply be a list/sequence of
            items (also untokenized)
        :return:
        """

        # if sequence is longer than max_seq_length - 1 (be able to append mask token) trim it
        # HINT: remove elements from beginning since we want to have the most recent history to
        # "predict the future"
        if type(data) is not list:
            raise ValueError("To prepare data for inference, please simply put in an unprocessed "
                             "sequence of data (i.e. a list of strings).")

        sequence = data[-cls.max_seq_len + 1:]

        # add [UNK] token to end of sequence (serves as a placeholder for inserting
        # the mask token, to be able to get the mlm_logits for this token for prediction purposes)
        # but basically any value may be added here
        sequence.append("[UNK]")

        cls.set_properties(
            tokenizer=cls.tokenizer,
            max_seq_len=cls.max_seq_len,
            max_predictions_per_seq=cls.max_predictions_per_seq,
            mask_token_id=cls.mask_token_id,
            unk_token_id=cls.unk_token_id,
            pad_token_id=cls.pad_token_id,
            masked_lm_rate=cls.masked_lm_rate,
            mask_token_rate=cls.mask_token_rate,
            random_token_rate=cls.random_token_rate
        )

        preprocessed_sequence = cls.process_element(sequence, True, True)

        # expand dimension of tensors since encoder needs inputs of dimension 2
        for key, value in preprocessed_sequence.items():
            if tf.is_tensor(value):
                preprocessed_sequence[key] = tf.expand_dims(value, axis=0)

        return preprocessed_sequence

    @classmethod
    def _ds_map_fn(cls, apply_mlm: bool, finetuning: bool, sequence):
        """
        See `call_process_element()` method, why this intermediate step is necessary.
        """

        processed_features = dict()

        if apply_mlm:
            output = tf.py_function(
                cls._call_process_element,
                [sequence, apply_mlm, finetuning],
                [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64]
            )
            processed_features["masked_lm_ids"] = output[3]
            processed_features["masked_lm_positions"] = output[4]
            processed_features["masked_lm_weights"] = output[5]
        else:
            output = tf.py_function(
                cls._call_process_element,
                [sequence, apply_mlm, finetuning],
                [tf.int64, tf.int64, tf.int64]
            )

        processed_features["labels"] = output[0]
        processed_features["input_word_ids"] = output[1]
        processed_features["input_mask"] = output[2]

        return processed_features

    @classmethod
    def _call_process_element(cls, sequence, apply_mlm: bool, finetuning: bool):
        """
        This function simply calls the `feature_preprocessing()` method and returns its return
        values as a list. This intermediate step allows the execution of python code in a
        tensorflow environment. Otherwise, the tokenizer could not alternate the given tensors
        during the `dataset.map()` procedure.
        """
        model_inputs = cls.process_element(sequence, apply_mlm, finetuning)

        # detailed declaration to ensure correct positions
        processed_features_list = [model_inputs["labels"], model_inputs["input_word_ids"], model_inputs["input_mask"]]

        if apply_mlm:
            processed_features_list.append(model_inputs["masked_lm_ids"])
            processed_features_list.append(model_inputs["masked_lm_positions"])
            processed_features_list.append(model_inputs["masked_lm_weights"])

        return processed_features_list
