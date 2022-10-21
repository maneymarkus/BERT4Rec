import random
import string


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
