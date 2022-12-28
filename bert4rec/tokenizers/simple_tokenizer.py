from absl import logging
from collections.abc import Iterable
import numbers
import os
import pandas as pd
import pathlib
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor, EagerTensor
import tqdm
from typing import Union

import bert4rec.tokenizers.base_tokenizer as base_tokenizer


class SimpleTokenizer(base_tokenizer.BaseTokenizer):
    """
    Converts a string to a unique (numerical) id
    """
    def __init__(self, vocab_file_path: pathlib.Path = None, extensible: bool = True):
        super().__init__(vocab_file_path=vocab_file_path, extensible=extensible)
        # each new entry is the key and the value is the corresponding token
        self._vocab = dict()

    @property
    def identifier(self):
        return "simple"

    def clear_vocab(self):
        self._vocab = dict()
        self._vocab_size = 0

    def tokenize(self, input, progress_bar: bool = False) -> Union[int, list[int], tf.RaggedTensor]:
        """
        This method tokenizes given input of different supported types and returns a tokenized string, list or
        dataframe column
        """
        if isinstance(input, bytes):
            input = input.decode()

        if isinstance(input, str):
            tokenized = self._tokenize_string(input)
        elif isinstance(input, tf.Tensor) or isinstance(input, tf.RaggedTensor):
            # tf.Tensor and tf.RaggedTensor have to be handled before the Iterable type as both are also Iterables
            tokenized = self._tokenize_tensor(input)
        elif isinstance(input, pd.Series):
            tokenized = self._tokenize_df_column(input)
        elif isinstance(input, Iterable):
            tokenized = self._tokenize_iterable(input, progress_bar)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return tokenized

    def detokenize(self,
                   token,
                   drop_tokens: list[str] = None,
                   progress_bar: bool = False) -> Union[int, list, pd.Series, tf.RaggedTensor]:
        """
        This method converts from tokens back to strings and returns either a detokenized string, list or
        dataframe column
        """
        if isinstance(token, numbers.Number):
            value = self._detokenize_token(token, drop_tokens)
        elif isinstance(token, tf.Tensor) or isinstance(token, tf.RaggedTensor):
            # tf.Tensor and tf.RaggedTensor have to be handled before the Iterable type as both are also Iterables
            value = self._detokenize_tensor(token, drop_tokens)
        elif isinstance(token, pd.Series):
            value = self._detokenize_df_column(token, drop_tokens)
        elif isinstance(token, Iterable):
            value = self._detokenize_iterable(token, drop_tokens, progress_bar)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return value

    def import_vocab_from_file(self, vocab_file: pathlib.Path) -> bool:
        if not vocab_file.is_file():
            raise RuntimeError(f'The vocab file does not exist (yet) or is not located at {vocab_file}.')

        logging.info(f"Importing vocab from file {vocab_file} (current vocab property is cleared).")
        self.clear_vocab()

        with open(vocab_file, "rb") as file:
            lines = file.readlines()
            if len(lines) <= 0:
                raise ValueError(f"The given vocab file ({vocab_file}) is empty.")
            if "|" not in lines[0].decode():
                raise ValueError(f"The given vocab file ({vocab_file}) does not contain "
                                 f"\"|\"-separated values.")
            if len(lines[0].decode().split("|")) != 2:
                raise ValueError(f"The given vocab file ({vocab_file}) should contain "
                                 f"\"|\"-separated key-value-pairs per individual line.")

            for line in lines:
                line = line.decode()
                line_parts = line.split("|")
                self._vocab[line_parts[0]] = int(line_parts[1])

        self._vocab_size = len(self._vocab.keys())

        return True

    def export_vocab_to_file(self, file_path: pathlib.Path) -> bool:
        if len(self._vocab.keys()) <= 0:
            raise ValueError("The vocab of the tokenizer is empty and therefore can't be written "
                             "to a file.")

        # generate comma seperated vocab file: key,token
        with open(file_path, "wb") as file:
            for key, token in self._vocab.items():
                line = key + "|" + str(token) + os.linesep
                line = bytes(line, "utf-8")
                file.write(line)

        return True


    def _tokenize_string(self, string: str) -> int:
        """
        Convert given string input to tokenizer-specific token

        :param string: String that should be converted to a token
        :return: Token
        """
        if isinstance(string, bytes):
            string = string.decode("utf-8")

        if string in self._vocab:
            token = self._vocab[string]
        else:
            if not self._extensible:
                raise RuntimeError(f"\"{string}\" is not known!")
            self._vocab[string] = self._vocab_size
            token = self._vocab_size
            self._vocab_size += 1

        return token

    def _tokenize_iterable(self, iterable: Iterable, progress_bar: bool = False) -> list[int]:
        """
        Convert a list of values to a list of representing tokens

        :param iterable: Iterable that should (individually) be converted to tokens
        :return: List of tokens
        """
        # usage of dict for performance reasons
        tokenized = dict()
        for i, token in enumerate(tqdm.tqdm(iterable) if progress_bar else iterable):
            tokenized[i] = self.tokenize(token)
        return list(tokenized.values())

    def _tokenize_df_column(self, df_column_input: pd.Series) -> pd.Series:
        """
        This method tokenizes a given dataframe column (from pandas)

        Alternative code with multiprocessing
        with mp.Pool(self.n_cores) as p:
            split = np.array_split(column, self.n_cores)
            p.map(self.__call__, split)
        """
        return df_column_input.map(self.tokenize)

    def _tokenize_tensor(self, tensor: [Tensor, EagerTensor, tf.RaggedTensor]) -> tf.RaggedTensor:
        value = tensor.numpy()
        if isinstance(value, Iterable):
            tokenized = [self.tokenize(v) for v in value]
        else:
            tokenized = self.tokenize(value)
        return tf.ragged.constant(tokenized, dtype=tf.int64)

    def _detokenize_token(self, token: int, drop_tokens: list[str] = None):
        if token < 0 or token > self._vocab_size:
            logging.warning(f"The given token {token} is not in the vocabulary.")
            value = None
        else:
            index = list(self._vocab.values()).index(token)
            value = list(self._vocab.keys())[index]
        if drop_tokens and value in drop_tokens:
            value = None
        return value

    def _detokenize_iterable(self,
                             tokens: Iterable[int],
                             drop_tokens: list[str] = None,
                             progress_bar: bool = False) -> list:
        values = list()
        for t in (tqdm.tqdm(tokens) if progress_bar else tokens):
            value = self.detokenize(t, drop_tokens)
            if value is not None:
                values.append(value)
        return values

    def _detokenize_tensor(self,
                           tokenized_tensor: Union[Tensor, EagerTensor, tf.RaggedTensor],
                           drop_tokens: list[str] = None) -> tf.RaggedTensor:
        tensor_values = tokenized_tensor.numpy().tolist()
        values = self.detokenize(tensor_values, drop_tokens)
        return tf.ragged.constant(values, dtype=tf.string)

    def _detokenize_df_column(self, token_column: pd.Series, drop_tokens: list[str] = None) -> pd.Series:
        return token_column.map(self.detokenize, drop_tokens)
