from absl import logging
import numpy as np
import numpy.testing
import pandas as pd
import random
import tensorflow as tf

import bert4rec.tokenizers as tokenizers
import tests.test_utils as utils


class SimpleTokenizersTest(tf.test.TestCase):
    def setUp(self) -> None:
        super(SimpleTokenizersTest, self).setUp()
        logging.set_verbosity(logging.DEBUG)
        self.tokenizer = tokenizers.get("simple")

    def tearDown(self):
        self.tokenizer = None

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
        words = utils.generate_random_word_list(min_word_length, size=vocab_size)

        # let tokenizer tokenize each word in list to build up vocabulary
        for word in words:
            tokenizer.tokenize(word)
        return words

    def test_instantiation_simple_tokenizer(self):
        self.assertIsNotNone(self.tokenizer,
                             f"The tokenizer factory does not create a simple tokenizer")
        self.assertIsInstance(self.tokenizer,
                              tokenizers.BaseTokenizer,
                              f"The created simple tokenizer does not inherit from the base tokenizer class.")
        self.assertIsInstance(self.tokenizer,
                              tokenizers.SimpleTokenizer,
                              f"The created simple tokenizer is not of type \"simple\". "
                              f"Expected: {tokenizers.SimpleTokenizer}, current value: {type(self.tokenizer)}")

    def test_string_tokenization(self):
        token = self.tokenizer.tokenize("test1")
        self.assertIs(token, 0,
                      f"The simple tokenizer should tokenize the first input with a \"0\". "
                      f"Expected: 0, current value: {token}")

    def test_string_detokenization(self):
        word = "test"
        token = self.tokenizer.tokenize(word)
        detokenized_token = self.tokenizer.detokenize(token)
        self.assertEqual(word, detokenized_token,
                         f"Detokenizing (converting a token ({token}) back to its type of value "
                         f"before the tokenization ({detokenized_token})) "
                         f"should actually return its original value ({word})")

    def test_tokenize_duplicates(self):
        input1 = "test1"
        input2 = "test1"
        token1 = self.tokenizer.tokenize(input1)
        token2 = self.tokenizer.tokenize(input2)
        self.assertEqual(token1, token2,
                         f"Tokenizing different input should output different tokens. "
                         f"Expected: input1 == input2 => token1 == token2 "
                         f"Current value: {input1} != {input2} => {token1} != {token2}")

    def test_tokenize_non_duplicates(self):
        input1 = "test1"
        input2 = "test2"
        token1 = self.tokenizer.tokenize(input1)
        token2 = self.tokenizer.tokenize(input2)
        self.assertNotEqual(token1, token2,
                            f"Tokenizing different input should output different tokens. "
                            f"Expected: input1 != input2 => token1 != token2 "
                            f"Current value: {input1} == {input2} => {token1} == {token2}")

    def test_tokenizing_vocab(self):
        simple_tokenizer = tokenizers.get("simple")
        words = self._fill_tokenizer_vocab(simple_tokenizer, 5, 1000)
        random_word = random.choice(words)
        expected_token = words.index(random_word)
        token = simple_tokenizer.tokenize(random_word)
        self.assertEqual(token, expected_token,
                         f"Tokenizing the same input (in a bigger vocabulary and in different places in the code)"
                         f" should output the same token. "
                         f"Expected: {expected_token}, current value: {token}")

    def test_detokenizing_vocab(self):
        words = self._fill_tokenizer_vocab(self.tokenizer, 5, 1000)
        random_word = random.choice(words)
        token = self.tokenizer.tokenize(random_word)
        detokenized_token = self.tokenizer.detokenize(token)
        self.assertEqual(random_word, detokenized_token,
                         f"Detokenizing a token should \"convert\" it back to its original value. "
                         f"Expected: {random_word}, current value: {detokenized_token}")

    def test_tokenize_list(self):
        words = utils.generate_random_word_list()
        tokenized_list = self.tokenizer.tokenize(words)
        self.assertIsInstance(tokenized_list, list,
                              f"Tokenizing a list with the simple tokenizer should return "
                              f"a (tokenized) list."
                              f"Expected type: list, current type: {type(tokenized_list)}")
        self.assertIsInstance(tokenized_list[0], int,
                              f"Tokenizing a list with the simple tokenizer should return "
                              f"a (tokenized) list of integers. "
                              f"Expected type: int, current type: {type(tokenized_list[0])}")
        random_word = random.choice(words)
        token = self.tokenizer.tokenize(random_word)
        expected_token = words.index(random_word)
        self.assertEqual(expected_token, token,
                         f"Individually tokenizing (Token: {token}) a word (Word: {random_word}) of a words list again,"
                         f" should return the same token (Expected token: {expected_token}).")

    def test_tokenize_multi_dimensional_list(self):
        md_list = [
            utils.generate_random_word_list() for _ in range(5)
        ]
        md_list_2 = utils.generate_random_word_list(size=20)
        md_list_2.extend(md_list)
        tokenized_md_list = self.tokenizer.tokenize(md_list)
        tokenized_md_list_2 = self.tokenizer.tokenize(md_list_2)
        self.assertEqual(self.tokenizer.get_vocab_size(), 520,
                         f"The length of the vocab should be 520 "
                         f"but actually is: {self.tokenizer.get_vocab_size()}")
        self.assertEqual(len(md_list), len(tokenized_md_list),
                         f"The first dimension of the tokenized multi dimensional list should have "
                         f"equally many items as the original multi dimensional list. \n"
                         f"Original list:\n{md_list}\n"
                         f"Tokenized list:\n{tokenized_md_list}")
        self.assertEqual(len(md_list_2), len(tokenized_md_list_2),
                         f"The first dimension of the tokenized multi dimensional list should have "
                         f"equally many items as the original multi dimensional list. \n"
                         f"Original list:\n{md_list_2}\n"
                         f"Tokenized list:\n{tokenized_md_list_2}")
        self.assertEqual(len(md_list[0]), len(tokenized_md_list[0]),
                         f"The second dimension of the tokenized multi dimensional list should have "
                         f"equally many items as the original multi dimensional list. \n"
                         f"First item of original list:\n{md_list[0]}\n"
                         f"First item of tokenized list:\n{tokenized_md_list[0]}")

    def test_detokenize_multi_dimensional_list(self):
        md_list = [
            utils.generate_random_word_list() for _ in range(5)
        ]
        md_list_2 = utils.generate_random_word_list(size=20)
        md_list_2.extend(md_list)
        tokenized_md_list = self.tokenizer.tokenize(md_list)
        tokenized_md_list_2 = self.tokenizer.tokenize(md_list_2)
        detokenized_md_list = self.tokenizer.detokenize(tokenized_md_list)
        detokenized_md_list_2 = self.tokenizer.detokenize(tokenized_md_list_2)
        self.assertEqual(md_list, detokenized_md_list,
                         f"Detokenizing a prior tokenized multi dimensional list "
                         f"should produce the original list again.\n"
                         f"Original list:\n{md_list}\n"
                         f"Detokenized list:\n{detokenized_md_list}")
        self.assertEqual(md_list_2, detokenized_md_list_2,
                         f"Detokenizing a prior tokenized multi dimensional list "
                         f"should produce the original list again.\n"
                         f"Original list:\n{md_list_2}\n"
                         f"Detokenized list:\n{detokenized_md_list_2}")

    def test_detokenize_list(self):
        words = self._fill_tokenizer_vocab(self.tokenizer)
        tokenized_list = self.tokenizer.tokenize(words)
        detokenized_list = self.tokenizer.detokenize(tokenized_list)
        self.assertIsInstance(detokenized_list, list,
                              f"Detokenizing a (tokenized) list with the simple tokenizer should return "
                              f"a list."
                              f"Expected type: list, current type: {type(detokenized_list)}")
        self.assertIsInstance(detokenized_list[0], str,
                              f"Detokenizing a (tokenized) list with the simple tokenizer should return "
                              f"a list of values with the original type (in this case: strings). "
                              f"Expected type: str, current type: {type(detokenized_list[0])}")
        self.assertEqual(words, detokenized_list,
                         f"The original list {words} should be equal to the detokenized list, "
                         f"but is: {detokenized_list}")
        random_token = random.choice(range(len(words) - 1))
        expected_value = words[random_token]
        value = self.tokenizer.detokenize(random_token)
        self.assertEqual(expected_value, value,
                         f"Individually detokenizing (Detokenized: {value}) a token (Token: {random_token})"
                         f" of a prior tokenized words list,"
                         f" should return the original value (Original value: {expected_value})")

    def test_tokenize_df_column(self):
        words = utils.generate_random_word_list()
        original_df = pd.DataFrame(words)
        tokenized_column = self.tokenizer.tokenize(original_df[0])
        self.assertIsInstance(tokenized_column, pd.Series,
                              f"A tokenized column should be of type pd.Series, "
                              f"but instead got: {type(tokenized_column)}")
        self.assertIsInstance(tokenized_column[0], np.int64,
                              f"The tokenized column should only consist of type int values "
                              f"but instead got: {type(tokenized_column[0])}")
        index = random.choice(range(len(original_df[0]) - 1))
        random_word = original_df.iloc[index, 0]
        token = self.tokenizer.tokenize(random_word)
        expected_token = tokenized_column[index]
        self.assertEqual(expected_token, token,
                         f"Individually tokenizing (Token: {token}) a word (Word: {random_word}) of a words list again,"
                         f" should return the same token (Expected token: {expected_token}).")

    def test_detokenize_df_column(self):
        words = self._fill_tokenizer_vocab(self.tokenizer)
        original_df = pd.DataFrame(words)
        tokenized_column = self.tokenizer.tokenize(original_df[0])
        detokenized_column = self.tokenizer.detokenize(tokenized_column)
        self.assertIsInstance(detokenized_column, pd.Series,
                              f"Detokenizing a (tokenized) column (pd.Series) with the simple tokenizer should return "
                              f"a column (pd.Series)."
                              f"Expected type: pd.Series, current type: {type(detokenized_column)}")
        self.assertIsInstance(detokenized_column[0], str,
                              f"Detokenizing a (tokenized) column (pd.Series) with the simple tokenizer should return "
                              f"a column (pd.Series) of values with the original type (in this case: strings). "
                              f"Expected type: str, current type: {type(detokenized_column[0])}")
        pd.testing.assert_series_equal(original_df[0], detokenized_column)
        random_token = random.choice(range(len(original_df[0]) - 1))
        expected_value = original_df.iloc[random_token, 0]
        value = self.tokenizer.detokenize(random_token)
        self.assertEqual(expected_value, value,
                         f"Individually detokenizing (Detokenized: {value}) a token (Token: {random_token})"
                         f" of a prior tokenized words column (pd.Series),"
                         f" should return the original value (Original value: {expected_value})")

    def test_tokenize_tensor(self):
        words = utils.generate_random_word_list(size=20)
        md_list = [
            utils.generate_random_word_list(size=(random.randint(5, 15))) for _ in range(5)
        ]
        tensor = tf.constant(words)
        ragged_tensor = tf.ragged.constant(md_list)
        tokenized_tensor = self.tokenizer.tokenize(tensor)
        tokenized_ragged_tensor = self.tokenizer.tokenize(ragged_tensor)
        self.assertEqual(len(tensor.numpy()), len(tokenized_tensor.numpy()),
                         f"The length of the original tensor ({len(tensor.numpy())}) "
                         f"and the length of tokenized tensor should be equal, "
                         f"but is: {len(tokenized_tensor.numpy())}")
        self.assertEqual(len(ragged_tensor.numpy()), len(tokenized_ragged_tensor.numpy()),
                         f"The length of the original tensor ({len(ragged_tensor.numpy())}) "
                         f"and the length of tokenized tensor should be equal, "
                         f"but is: {len(tokenized_ragged_tensor.numpy())}")

    def test_detokenize_tensor(self):
        words = utils.generate_random_word_list(size=20)
        md_list = [
            utils.generate_random_word_list(size=(random.randint(5, 15))) for _ in range(5)
        ]
        tensor = tf.constant(words)
        ragged_tensor = tf.ragged.constant(md_list)
        tokenized_tensor = self.tokenizer.tokenize(tensor)
        tokenized_ragged_tensor = self.tokenizer.tokenize(ragged_tensor)
        detokenized_tensor = self.tokenizer.detokenize(tokenized_tensor)
        detokenized_ragged_tensor = self.tokenizer.detokenize(tokenized_ragged_tensor)
        numpy.testing.assert_array_equal(tensor.numpy(), detokenized_tensor.numpy(),
                                         f"The detokenized tensor should be equal to the original tensor.\n"
                                         f"Original tensor:\n{tensor}\n"
                                         f"Detokenized tensor:\n{detokenized_tensor}\n")
        ragged_tensor_values = ragged_tensor.numpy()
        detokenized_ragged_tensor_values = detokenized_ragged_tensor.numpy()
        for i in range(len(ragged_tensor_values) - 1):
            numpy.testing.assert_array_equal(ragged_tensor_values[i], detokenized_ragged_tensor_values[i],
                                             f"The detokenized ragged tensor should be equal "
                                             f"to the original ragged tensor.\n"
                                             f"Original tensor:\n{ragged_tensor}\n"
                                             f"Detokenized tensor:\n{detokenized_ragged_tensor}\n")

    def test_drop_tokens(self):
        words = self._fill_tokenizer_vocab(self.tokenizer)
        dt1 = "Token1"
        dt2 = "Token2"
        drop_tokens = [dt1, dt2]
        self.tokenizer.tokenize(drop_tokens)
        original_input = [random.choice(words) for _ in range(10)]
        original_input.append(dt1)
        original_input.append(dt2)
        tokenized = self.tokenizer.tokenize(original_input)
        detokenized = self.tokenizer.detokenize(tokenized, drop_tokens)
        self.assertEqual(original_input[:-2], detokenized,
                         f"Detokenizing some tokenized input ({original_input}) "
                         f"and dropping a list of drop_tokens ({drop_tokens}) "
                         f"should actually return the original input without the drop_tokens "
                         f"(so: {original_input[:-2]}) but actually returned: {detokenized}")

    def test_export_vocab(self):
        pass

    def test_import_vocab(self):
        pass


if __name__ == "__main__":
    tf.test.main()
