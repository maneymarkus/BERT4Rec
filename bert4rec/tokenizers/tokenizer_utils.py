import pathlib


def export_num_vocab_to_file(file_path: pathlib.Path, vocab: list) -> bool:
    """
    Exports a given `vocab` (is of type list) to a specified `file_path`.
    Each line represents a single entry of the vocab and the "line number" depicts
    the representing token.

    :param file_path: Location of vocab file to be put in
    :param vocab: Vocab list to be written to the filesystem
    :return: Boolean (True when everything was written to the file)
    """
    if len(vocab) <= 0:
        raise ValueError("The vocab that should be written to a file should have at least one entry!")

    with open(file_path, "w", encoding="utf-8") as file:
        for token in vocab:
            print(token, file=file)

    return True


def import_num_vocab_from_file(file_path: pathlib.Path) -> list[str]:
    """
    Reads a vocab file located at `file_path` into a list of strings and returns it.

    :param file_path: The location of the file to read the vocab from
    :return: Vocab list
    """
    if not file_path.is_file():
        raise RuntimeError(f'The vocab file does not exist (yet) or is not located at {file_path}.')

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        vocab = [line.strip() for line in lines]
        file.close()

    return vocab
