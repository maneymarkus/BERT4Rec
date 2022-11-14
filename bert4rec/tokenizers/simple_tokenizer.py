from collections.abc import Iterable
import numbers
import pandas as pd
import pathlib
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor, EagerTensor
from typing import Union

import bert4rec.tokenizers.base_tokenizer as base_tokenizer
import bert4rec.tokenizers.tokenizer_utils as utils


class SimpleTokenizer(base_tokenizer.BaseTokenizer):
    """
    Converts a string to a unique (numerical) id
    E.g. a given string containing any potential delimiting/splitting symbols (like "," or " " or "|") is converted
    to a list of ONE id.
    Even more concrete example: "Action|Drama|Children" => [2]
    See: MultiHotContentTokenizer for comparison
    """
    def __init__(self, vocab_file_path: pathlib.Path = None, extensible: bool = True):
        """
        Converts a string containing multiple properties seperated by a specific symbol
        into a list of ids, where each id represents one property (so each string
        contains multiple part strings seperated by a symbol)
        E.g. a string containing categories is converted to a list of ids
        Even more concrete example: "Action|Drama|Children" => [3,5,6]
        See: FullStringToOneIdContentTokenizer for comparison
        """
        super().__init__(vocab_file_path=vocab_file_path, extensible=extensible)
        # initialize token map as list as this tokenizer uses numerical tokens and increases the token for each new
        # entry
        self._vocab = list()

    @property
    def identifier(self):
        return "simple"

    def clear_vocab(self):
        self._vocab = list()
        self._vocab_size = 0

    def tokenize(self, input) -> Union[int, list[int], tf.RaggedTensor]:
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
            tokenized = self._tokenize_iterable(input)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return tokenized

    def detokenize(self,
                   token,
                   drop_tokens: list[str] = None) -> Union[int, list, pd.Series, tf.RaggedTensor]:
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
            value = self._detokenize_iterable(token, drop_tokens)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return value

    def import_vocab_from_file(self, vocab_file: pathlib.Path) -> None:
        self._vocab = utils.import_num_vocab_from_file(vocab_file)
        self._vocab_size = len(self._vocab)

    def export_vocab_to_file(self, file_path: pathlib.Path) -> None:
        utils.export_num_vocab_to_file(file_path, self._vocab)

    def _tokenize_string(self, string: str) -> int:
        """
        Convert given string input to tokenizer-specific token

        :param string: String that should be converted to a token
        :return: Token
        """
        if isinstance(string, bytes):
            string = string.decode("utf-8")

        if string in self._vocab:
            token = self._vocab.index(string)
        else:
            if not self._extensible:
                raise RuntimeError(f"\"{string}\" is not known!")
            self._vocab.append(string)
            self._vocab_size = len(self._vocab)
            token = self._vocab_size - 1

        return token

    def _tokenize_iterable(self, iterable: Iterable) -> list[int]:
        """
        Convert a list of values to a list of representing tokens

        :param iterable: Iterable that should (individually) be converted to tokens
        :return: List of tokens
        """
        tokenized = list()
        for token in iterable:
            tokenized.append(self.tokenize(token))
        return tokenized

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
        value = None
        if 0 <= token < self._vocab_size:
            value = self._vocab[token]
        if drop_tokens and value in drop_tokens:
            value = None
        return value

    def _detokenize_iterable(self, tokens: Iterable[int], drop_tokens: list[str] = None) -> list:
        values = list()
        for t in tokens:
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
