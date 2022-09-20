import json
import pathlib
import tensorflow as tf

from bert4rec.utils import utils


def determine_model_path(path: pathlib.Path, mode: int = 0) -> pathlib.Path:
    """
    Determines the path for loading or storing a ml model depending on the given mode.

    :param path:
    :param mode: The mode determines the base path of the given path. Available modes are 0, 1, 2.
    0 is default and adds the project root to the path. 1 adds the virtual environment to the path
    and 2 adds the current working directory to the path (i.e. doesn't add anything to the path).
    Modes 0 and 1 also use predefined directories (saved_models).
    :return:
    """
    if path.is_absolute():
        return path

    if mode == 0:
        # path relative to project root
        determined_path = utils.get_project_root()\
            .joinpath(utils.get_default_model_save_path())\
            .joinpath(path)
    elif mode == 1:
        # path relative to virtual environment
        determined_path = utils.get_virtual_env_path() \
            .joinpath(utils.get_default_model_save_path()) \
            .joinpath(path)
    elif mode == 2:
        # path not altered (relative to current working directory
        determined_path = path
    else:
        raise ValueError(f"The mode parameter has to be in the range of [0, 1, 2], but is: {mode}")

    return determined_path


def rank_items(logits: tf.Tensor, embeddings: tf.Tensor, items: list):
    """
    Ranks a given list of items based on their given vocab embeddings and their machine learning model
    output logits

    :param logits: ML model output logits
    :param embeddings: The gathered embeddings of the items to be ranked
    :param items: The items that should be ranked
    :return: A tuple with the first element containing the (unsorted) probabilities of the items and the
    second element containing the ranked elements in descending order (highest probability first)
    """
    assert len(logits) == len(tf.transpose(embeddings)), \
        f"The length of the logits tensor should be equal to the length of the first dimension of the " \
        f"transposed embeddings tensor"

    assert len(embeddings) == len(items), \
        f"The length of the embeddings tensor should be equal to the length of the list of items to be ranked, " \
        f"as each item should have an embedding."

    vocab_logits = tf.einsum("n,nm->m", logits, tf.transpose(embeddings))
    vocab_probabilities = tf.nn.softmax(vocab_logits)
    sorted_indexes = tf.argsort(vocab_probabilities, direction="DESCENDING")
    ranking = tf.gather(items, sorted_indexes)
    return vocab_probabilities, ranking


def load_config(save_path: pathlib.Path) -> dict:
    """
    Loads a json config into a python dict. To use the dict as a parameter in a function call instead
    of positional parameters make use of the double asteriks (**)

    :param save_path:
    :return:
    """
    if not save_path.is_file():
        raise FileNotFoundError(f"No config file exists at given path: {save_path}")

    with open(save_path, "r") as jf:
        config = json.load(jf)

    return config


if __name__ == "__main__":
    path = pathlib.Path("my_model")
    determined_path_1 = determine_model_path(path, 0)
    print(determined_path_1)
    determined_path_2 = determine_model_path(path, 1)
    print(determined_path_2)
    determined_path_3 = determine_model_path(path, 2)
    print(determined_path_3)

