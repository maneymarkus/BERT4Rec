import random
import string
import tensorflow as tf

from bert4rec.dataloaders import samplers


def generate_random_word_list(min_word_length: int = 5,
                              max_word_length: int = 25,
                              size: int = 100,
                              seed: int = None) -> list[str]:
    """
    Generate a list of length `vocab_size` with random words with a minimum length of
    `min_word_length`. Parameter `seed` can be given for reproducible results

    :param min_word_length:
    :param max_word_length:
    :param size:
    :param seed:
    :return:
    """
    random.seed(seed)

    # generate random word list
    words = [
        "".join(
            random.choice(string.ascii_letters) for _ in range(
                random.randint(min_word_length, max_word_length)
            )
        ) for _ in range(size)
    ]

    # remove duplicates
    words = list(dict.fromkeys(words))
    return words


def generate_random_sequence_dataset(ds_size: int = 1000,
                                     min_seq_len: int = 5,
                                     max_seq_len: int = 100,
                                     vocab_size: int = 1000,
                                     seed: int = None) -> tf.data.Dataset:
    random.seed(seed)

    vocab = generate_random_word_list(size=vocab_size, seed=seed)
    sampler = samplers.RandomSampler()

    sequence_list = []
    for i in range(ds_size):
        sequence_length = random.randint(min_seq_len, max_seq_len)
        sequence = sampler.sample(sequence_length, vocab=vocab, allow_duplicates=True)
        sequence_list.append(sequence)
    sequences = tf.ragged.constant(sequence_list)
    ds = tf.data.Dataset.from_tensor_slices(sequences)

    return ds
