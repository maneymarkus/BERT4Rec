import numpy as np
import pandas as pd
import random
import unittest

import bert4rec.tokenizers as tokenizers
import tests.test_utils as utils


class SimpleTokenizersTest(unittest.TestCase):
    def _fill_tokenizer_vocab(self,
                              tokenizer: tokenizers.BaseTokenizer,
                              min_word_length: int = 5,
                              vocab_size: int = 100):
        """
        Fill the vocab of a given `tokenizer` with random words

        :param tokenizer:
        :param min_word_length:
        :param vocab_size:
        :return:
        """
        # generate random word list
        words = utils.generate_unique_word_list(min_word_length, vocab_size)

        # let tokenizer tokenize each word in list to build up vocabulary
        for word in words:
            tokenizer.tokenize(word)
        return words

    def test_instantiation_simple_tokenizer(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        self.assertIsNotNone(simple_tokenizer,
                             f"The tokenizer factory does not create a simple tokenizer")
        self.assertIsInstance(simple_tokenizer,
                              tokenizers.BaseTokenizer,
                              f"The created simple tokenizer does not inherit from the base tokenizer class.")
        self.assertIsInstance(simple_tokenizer,
                              tokenizers.SimpleTokenizer,
                              f"The created simple tokenizer is not of type \"simple\". "
                              f"Expected: {tokenizers.SimpleTokenizer}, current value: {type(simple_tokenizer)}")

    def test_string_tokenization(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        token = simple_tokenizer.tokenize("test1")
        self.assertIs(token, 0,
                      f"The simple tokenizer should tokenize the first input with a \"0\". "
                      f"Expected: 0, current value: {token}")

    def test_string_detokenization(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        word = "test"
        token = simple_tokenizer.tokenize(word)
        detokenized_token = simple_tokenizer.detokenize(token)
        self.assertEqual(word, detokenized_token,
                         f"Detokenizing (converting a token ({token}) back to its type of value "
                         f"before the tokenization ({detokenized_token})) "
                         f"should actually return its original value ({word})")

    def test_tokenize_duplicates(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        input1 = "test1"
        input2 = "test1"
        token1 = simple_tokenizer.tokenize(input1)
        token2 = simple_tokenizer.tokenize(input2)
        self.assertEqual(token1, token2,
                         f"Tokenizing different input should output different tokens. "
                         f"Expected: input1 == input2 => token1 == token2 "
                         f"Current value: {input1} != {input2} => {token1} != {token2}")

    def test_tokenize_non_duplicates(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        input1 = "test1"
        input2 = "test2"
        token1 = simple_tokenizer.tokenize(input1)
        token2 = simple_tokenizer.tokenize(input2)
        self.assertNotEqual(token1, token2,
                            f"Tokenizing different input should output different tokens. "
                            f"Expected: input1 != input2 => token1 != token2 "
                            f"Current value: {input1} == {input2} => {token1} == {token2}")

    def test_tokenizing_vocab(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = self._fill_tokenizer_vocab(simple_tokenizer, 5, 1000)
        random_word = random.choice(words)
        expected_token = words.index(random_word)
        token = simple_tokenizer.tokenize(random_word)
        self.assertEqual(token, expected_token,
                         f"Tokenizing the same input (in a bigger vocabulary and in different places in the code)"
                         f" should output the same token. "
                         f"Expected: {expected_token}, current value: {token}")

    def test_detokenizing_vocab(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = self._fill_tokenizer_vocab(simple_tokenizer, 5, 1000)
        random_word = random.choice(words)
        token = simple_tokenizer.tokenize(random_word)
        detokenized_token = simple_tokenizer.detokenize(token)
        self.assertEqual(random_word, detokenized_token,
                         f"Detokenizing a token should \"convert\" it back to its original value. "
                         f"Expected: {random_word}, current value: {detokenized_token}")

    def test_tokenize_list(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = utils.generate_unique_word_list()
        tokenized_list = simple_tokenizer.tokenize(words)
        self.assertIsInstance(tokenized_list, list,
                              f"Tokenizing a list with the simple tokenizer should return "
                              f"a (tokenized) list."
                              f"Expected type: list, current type: {type(tokenized_list)}")
        self.assertIsInstance(tokenized_list[0], int,
                              f"Tokenizing a list with the simple tokenizer should return "
                              f"a (tokenized) list of integers. "
                              f"Expected type: int, current type: {type(tokenized_list[0])}")
        random_word = random.choice(words)
        token = simple_tokenizer.tokenize(random_word)
        expected_token = words.index(random_word)
        self.assertEqual(expected_token, token,
                         f"Individually tokenizing (Token: {token}) a word (Word: {random_word}) of a words list again,"
                         f" should return the same token (Expected token: {expected_token}).")

    def test_detokenize_list(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = self._fill_tokenizer_vocab(simple_tokenizer)
        tokenized_list = simple_tokenizer.tokenize(words)
        detokenized_list = simple_tokenizer.detokenize(tokenized_list)
        self.assertIsInstance(detokenized_list, list,
                              f"Detokenizing a (tokenized) list with the simple tokenizer should return "
                              f"a list."
                              f"Expected type: list, current type: {type(detokenized_list)}")
        self.assertIsInstance(detokenized_list[0], str,
                              f"Detokenizing a (tokenized) list with the simple tokenizer should return "
                              f"a list of values with the original type (in this case: strings). "
                              f"Expected type: str, current type: {type(detokenized_list[0])}")
        random_token = random.choice(range(len(words) - 1))
        expected_value = words[random_token]
        value = simple_tokenizer.detokenize(random_token)
        self.assertEqual(expected_value, value,
                         f"Individually detokenizing (Detokenized: {value}) a token (Token: {random_token})"
                         f" of a prior tokenized words list,"
                         f" should return the original value (Original value: {expected_value})")

    def test_tokenize_df_column(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = utils.generate_unique_word_list()
        original_df = pd.DataFrame(words)
        tokenized_column = simple_tokenizer.tokenize(original_df[0])
        self.assertIsInstance(tokenized_column, pd.Series,
                              f"A tokenized column should be of type pd.Series, "
                              f"but instead got: {type(tokenized_column)}")
        self.assertIsInstance(tokenized_column[0], np.int64,
                              f"The tokenized column should only consist of type int values "
                              f"but instead got: {type(tokenized_column[0])}")
        index = random.choice(range(len(original_df[0]) - 1))
        random_word = original_df.iloc[index, 0]
        token = simple_tokenizer.tokenize(random_word)
        expected_token = tokenized_column[index]
        self.assertEqual(expected_token, token,
                         f"Individually tokenizing (Token: {token}) a word (Word: {random_word}) of a words list again,"
                         f" should return the same token (Expected token: {expected_token}).")

    def test_detokenize_df_column(self):
        simple_tokenizer = tokenizers.tokenizer_factory.get_tokenizer("simple")
        words = self._fill_tokenizer_vocab(simple_tokenizer)
        original_df = pd.DataFrame(words)
        tokenized_column = simple_tokenizer.tokenize(original_df[0])
        detokenized_column = simple_tokenizer.detokenize(tokenized_column)
        self.assertIsInstance(detokenized_column, pd.Series,
                              f"Detokenizing a (tokenized) column (pd.Series) with the simple tokenizer should return "
                              f"a column (pd.Series)."
                              f"Expected type: pd.Series, current type: {type(detokenized_column)}")
        self.assertIsInstance(detokenized_column[0], str,
                              f"Detokenizing a (tokenized) column (pd.Series) with the simple tokenizer should return "
                              f"a column (pd.Series) of values with the original type (in this case: strings). "
                              f"Expected type: str, current type: {type(detokenized_column[0])}")
        random_token = random.choice(range(len(original_df[0]) - 1))
        expected_value = original_df.iloc[random_token, 0]
        value = simple_tokenizer.detokenize(random_token)
        self.assertEqual(expected_value, value,
                         f"Individually detokenizing (Detokenized: {value}) a token (Token: {random_token})"
                         f" of a prior tokenized words column (pd.Series),"
                         f" should return the original value (Original value: {expected_value})")

    def test_export_vocab(self):
        pass

    def test_import_vocab(self):
        pass


if __name__ == "__main__":
    unittest.main()
