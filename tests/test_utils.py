import random
import string


def generate_unique_word_list(min_word_length: int = 5, vocab_size: int = 100) -> list[str]:
    """
    Generate a list of length `vocab_size` with random words with a minimum length of
    `min_word_length`

    :param min_word_length:
    :param vocab_size:
    :return:
    """
    # generate random word list
    words = [
        "".join(
            random.choice(string.ascii_letters) for _ in range(
                random.randint(min_word_length, min_word_length + random.randint(0, 10))
            )
        ) for _ in range(vocab_size)
    ]

    # remove duplicates
    words = list(dict.fromkeys(words))
    return words
