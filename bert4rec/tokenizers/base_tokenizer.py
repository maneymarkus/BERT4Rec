import abc
import pathlib


class BaseTokenizer(abc.ABC):
    """
    Converts given strings/lists/dataframe_columns/tensors to ids.
    The token_store contains the vocab and maps tokens to a single entry of the vocab. The token_store class variable
    should be initialised by each concrete tokenizer. It could be a simple list/array with the indexes of the entries
    depicting the corresponding tokens or if non-numerical tokens are used it should be a dictionary mapping
    tokens to entries.

    Usage:
    1. Instantiate a concrete tokenizer
    2. Call the self.tokenize() method (See method for supported argument types).
        This method will generate a dictionary mapping strings to tokens
    3. Retrieve single tokens via the self.get_token() method or convert a whole dataframe column
        to tokens via the self.convert_column_to_ids() method
    """
    def __init__(self, vocab_file_path: pathlib.Path = None):
        self.max_len = 1
        self.vocab_size = 0
        self.vocab = None
        if vocab_file_path is not None and vocab_file_path.is_file():
            self.import_vocab_from_file(vocab_file_path)

    @abc.abstractmethod
    def clear_vocab(self):
        pass

    @abc.abstractmethod
    def tokenize(self, input):
        """
        Convert (virtually any) input to a tokenizer-specific token.
        To support multiple input types, this method may call specific private methods
        like `_tokenize_string(string)` to tokenize specific input.

        :param input: To be converted
        :return: Token (may be other string or integer)
        """
        pass

    @abc.abstractmethod
    def detokenize(self, token: str):
        """
        Return the corresponding string the given `token` represents

        :param token: Should be converted back to original value
        :return: Corresponding (original) value
        """
        pass

    @abc.abstractmethod
    def export_vocab_to_file(self, file_path: pathlib.Path):
        """
        Export current vocab to a file to given `file_path`

        :param file_path: The location the vocab file should be put
        :return: None
        """
        pass

    @abc.abstractmethod
    def import_vocab_from_file(self, file_path: pathlib.Path):
        """
        Import and initialise vocab from a file at the given `file_path`

        :param file_path: The location of the vocab file
        :return: None
        """
        pass

    @property
    def max_seq_len(self) -> int:
        """
        Maximal number of possible item collections in the self.token_dict.
        The value is used for static padding.
        See multi_hot_tokenizer.
        """
        return self.max_len

    def get_vocab(self) -> dict:
        return self.vocab

    def get_vocab_size(self) -> int:
        return self.vocab_size
