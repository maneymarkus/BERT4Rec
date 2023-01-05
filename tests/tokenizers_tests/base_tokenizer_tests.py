import random
import tensorflow as tf

import bert4rec.tokenizers as tokenizers
import tests.test_utils as utils


class BaseTokenizerTests(tf.test.TestCase):
    def setUp(self):
        super(BaseTokenizerTests, self).setUp()
        # Testing base tokenizer features with a concrete tokenizer implementation (as abstract classes
        # can't be instantiated)
        self.tokenizer = tokenizers.get("simple")
        pass

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

    def test_tokenizer_factory_method(self):
        tokenizer = tokenizers.get("simple")
        self.assertIsInstance(tokenizer, tokenizers.SimpleTokenizer)
        tokenizer2 = tokenizers.get()
        self.assertIsInstance(tokenizer2, tokenizers.SimpleTokenizer)
        with self.assertRaises(ValueError):
            tokenizers.get("lkfsahdkjlfh")

    def test_switching_extensibility(self):
        words = self._fill_tokenizer_vocab(self.tokenizer, min_word_length=10)
        self.tokenizer.disable_extensibility()
        new_word = "Token"
        with self.assertRaises(RuntimeError):
            self.tokenizer.tokenize(new_word)

        self.tokenizer.enable_extensibility()
        tokenized_new_word = self.tokenizer.tokenize(new_word)
        self.assertEqual(tokenized_new_word, len(words),
                         f"After enabling the extensibility again, tokenizing a new word ({new_word}) should "
                         f"tokenize it and thereby dynamically adding it to the vocab.")

    def test_get_vocab(self):
        vocab_size = random.randint(50, 150)
        vocab = self.tokenizer.get_vocab()
        self.assertIsNotNone(vocab, "The vocab object returned by the tokenizer should be not None "
                                    "after initializing it but actually is.")
        self.assertEmpty(vocab, "The vocab object returned by the tokenizer should be empty "
                                "prior to filling it.")
        self._fill_tokenizer_vocab(self.tokenizer, vocab_size=vocab_size)
        vocab = self.tokenizer.get_vocab()
        self.assertIsNotNone(vocab,
                             f"The vocab returned by the tokenizer should not be `None` after filling the tokenizer")
        self.assertEqual(len(vocab), vocab_size,
                         f"The vocab returned by the tokenizer should have a length of {vocab_size} but "
                         f"actually has a length of: {len(vocab)}")

    def test_get_vocab_size(self):
        vocab_size = random.randint(50, 150)
        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(tokenizer_vocab_size, 0,
                         f"The vocab size returned by the tokenizer should be an 0 prior to filling it,"
                         f"but actually is: {tokenizer_vocab_size}")
        self._fill_tokenizer_vocab(self.tokenizer, vocab_size=vocab_size)
        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(tokenizer_vocab_size, vocab_size,
                         f"The vocab size returned by the tokenizer should be {vocab_size} but "
                         f"actually is: {tokenizer_vocab_size}")


if __name__ == '__main__':
    tf.test.main()
