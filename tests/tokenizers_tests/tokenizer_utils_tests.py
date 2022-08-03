from absl import logging
import pathlib
import random
import tempfile
import tensorflow as tf
import uuid

import tests.test_utils as utils
import bert4rec.tokenizers.tokenizer_utils as tokenizer_utils


class TokenizerUtilsTest(tf.test.TestCase):
    def setUp(self):
        super(TokenizerUtilsTest, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def _create_vocab_file(self, vocab_size: int = 100):
        vocab = utils.generate_unique_word_list(size=vocab_size)
        tmp = tempfile.NamedTemporaryFile(mode="wt", delete=False)
        for word in vocab:
            tmp.write(word + "\n")
        tmp.close()
        return vocab, tmp

    def test_export_num_vocab_to_file(self):
        with self.assertRaises(ValueError):
            _ = tokenizer_utils.export_num_vocab_to_file(pathlib.Path("/" + str(uuid.uuid4())), [])

        vocab_size = 25
        vocab = utils.generate_unique_word_list(size=vocab_size)
        logging.debug("\nVocab:\n" + str(vocab))
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        path = pathlib.Path(tmp.name)
        result = tokenizer_utils.export_num_vocab_to_file(path, vocab)
        self.assertEqual(result, True,
                         f"The export function should return True on success.")

        random_indexes = [random.randint(0, vocab_size - 1) for _ in range(round(vocab_size * 0.25))]
        random_indexes.sort()
        logging.debug(random_indexes)
        index = random_indexes.pop(0)
        with open(path, "r", encoding="utf-8") as vf:
            line_counter = 1
            for line in vf:
                if index == (line_counter - 1):
                    word_from_file = line.strip()
                    word_from_vocab = vocab[index]
                    self.assertEqual(word_from_file, word_from_vocab,
                                     f"The word at index {index} in the vocab ({word_from_vocab}) "
                                     f"should be equal to the word at line {line_counter}, "
                                     f"but is actually: {word_from_file}")
                    if random_indexes:
                        index = random_indexes.pop(0)
                line_counter += 1

    def test_import_num_vocab_from_file(self):
        original_vocab, vocab_file = self._create_vocab_file(vocab_size=25)

        random_path = pathlib.Path("/random/" + str(uuid.uuid4()))
        with self.assertRaises(RuntimeError):
            _ = tokenizer_utils.import_num_vocab_from_file(random_path)

        vocab = tokenizer_utils.import_num_vocab_from_file(pathlib.Path(vocab_file.name))
        self.assertEqual(original_vocab, vocab, f"The original vocab list:\n{original_vocab}\n "
                                                f"should be equal to the read vocab from the vocab file, "
                                                f"but actually is:\n{vocab}")


if __name__ == "__main__":
    tf.test.main()
