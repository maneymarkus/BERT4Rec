import random
import string
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader, BERT4RecDataloader


def generate_random_word_list(min_word_length: int = 5,
                              max_word_length: int = 25,
                              size: int = 100) -> list[str]:
    """
    Generate a list of length `vocab_size` with random words with a minimum length of
    `min_word_length`

    :param min_word_length:
    :param max_word_length:
    :param size:
    :return:
    """
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
                                     seq_min_len: int = 5,
                                     seq_max_len: int = 100,
                                     dataloader: BaseDataloader = None) -> (tf.data.Dataset, BaseDataloader):
    subject_list = []
    sequence_list = []
    for i in range(ds_size):
        subject_list.append(random.randint(0, ds_size * 2))
        sequence_length = random.randint(seq_min_len, seq_max_len)
        sequence = generate_random_word_list(size=sequence_length)
        sequence_list.append(sequence)
    sequences = tf.ragged.constant(sequence_list)
    ds = tf.data.Dataset.from_tensor_slices((subject_list, sequences))

    dataloader = BERT4RecDataloader(max_seq_len=seq_max_len, max_predictions_per_seq=5)

    dataloader.generate_vocab(sequences)
    prepared_ds = dataloader.preprocess_dataset(ds, finetuning=True)
    return prepared_ds, dataloader

