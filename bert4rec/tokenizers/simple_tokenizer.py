from collections.abc import Iterable
import pandas as pd
import pathlib
import tensorflow as tf
from typing import Union

import bert4rec.tokenizers.base_tokenizer as base_tokenizer
import bert4rec.tokenizers.tokenizer_utils as utils


class SimpleTokenizer(base_tokenizer.BaseTokenizer):
    """
    Converts a string to a unique (numerical) id
    E.g. a given string containing any potential delimiting/splitting symbols (like "," or " " or "|" is converted
    to a list of ONE id.
    Even more concrete example: "Action|Drama|Children" => [2]
    See: MultiHotContentTokenizer for comparison
    """
    def __init__(self, vocab_file_path: pathlib.Path = None):
        """
        Converts a string containing multiple properties seperated by a specific symbol
        into a list of ids, where each id represents one property (so each string
        contains multiple part strings seperated by a symbol)
        E.g. a string containing categories is converted to a list of ids
        Even more concrete example: "Action|Drama|Children" => [3,5,6]
        See: FullStringToOneIdContentTokenizer for comparison
        """
        super().__init__(vocab_file_path=vocab_file_path)
        # initialize token map as list as this tokenizer uses numerical tokens and increases the token for each new
        # entry
        self.vocab = list()

    def clear_vocab(self):
        self.vocab = list()
        self.vocab_size = 0

    def tokenize(self, input: Union[str, Iterable, pd.Series]) -> Union[int, list[int]]:
        """
        This method tokenizes given input of different supported types and returns a tokenized string, list or
        dataframe column
        """
        if isinstance(input, str):
            tokenized = self._tokenize_string(input)
        elif isinstance(input, tf.Tensor) or isinstance(input, tf.RaggedTensor):
            # since tf.Tensor and tf.RaggedTensor are iterable there is no need to specify them as an allowed input type
            # separately. However, they need to be handled differently
            tokenized = self._tokenize_tensor(input)
        elif isinstance(input, pd.Series):
            tokenized = self._tokenize_df_column(input)
        elif isinstance(input, Iterable):
            tokenized = self._tokenize_iterable(input)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return tokenized

    def detokenize(self, token: Union[int, Iterable[int], pd.Series]) -> Union[int, list, pd.Series]:
        """
        This method converts from tokens back to strings and returns either a detokenized string, list or
        dataframe column
        """
        if isinstance(token, int):
            value = self._detokenize_token(token)
        elif isinstance(token, pd.Series):
            value = self._detokenize_df_column(token)
        elif isinstance(token, Iterable):
            value = self._detokenize_iterable(token)
        else:
            raise ValueError("The provided argument is not of a supported type")
        return value

    def import_vocab_from_file(self, vocab_file: pathlib.Path) -> None:
        self.vocab = utils.import_num_vocab_from_file(vocab_file)

    def export_vocab_to_file(self, file_path: pathlib.Path) -> None:
        utils.export_num_vocab_to_file(file_path, self.vocab)

    def _tokenize_string(self, string: str) -> int:
        """
        Convert given string input to tokenizer-specific token

        :param string: String that should be converted to a token
        :return: Token
        """
        if isinstance(string, bytes):
            string = string.decode("utf-8")

        if string in self.vocab:
            token = self.vocab.index(string)
        else:
            self.vocab.append(string)
            self.vocab_size = len(self.vocab)
            token = self.vocab_size - 1
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
        return df_column_input.map(self._tokenize_string)

    def _tokenize_tensor(self, tensor: [tf.Tensor, tf.RaggedTensor]) -> tf.Tensor:
        tokenized = list()
        value_list = tensor.numpy()
        tokenized = [self._tokenize_string(v) for v in value_list]
        return tf.ragged.constant([tokenized], dtype=tf.int64)

    def _detokenize_token(self, token: int):
        value = None
        if 0 <= token < self.vocab_size:
            value = self.vocab[token]
        return value

    def _detokenize_iterable(self, tokens: Iterable[int]) -> list:
        values = list()
        for t in tokens:
            values.append(self._detokenize_token(t))
        return values

    def _detokenize_df_column(self, token_column: pd.Series) -> pd.Series:
        return token_column.map(self._detokenize_token)
