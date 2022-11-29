import abc
import pathlib
import tensorflow as tf


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
    def __init__(self,
                 vocab_file_path: pathlib.Path = None,
                 extensible: bool = True):
        """
        Initializes a tokenizer inheriting from this base class

        :param vocab_file_path: The path to the vocab file
        :param extensible: Determines the mode in which this tokenizer works. If true, then newly encountered
        items will be dynamically tokenized with a new token. If false, unknown tokens will throw an error
        """
        self._max_len = 1
        self._vocab_size = 0
        self._vocab = None
        self._extensible = extensible
        if vocab_file_path is not None and vocab_file_path.is_file():
            self._extensible = False
            self.import_vocab_from_file(vocab_file_path)

    def generate_vocab_from_ds(self, ds: tf.data.Dataset):
        """
        Generate vocab of the tokenizer by traversing the given text_ds.
        May take some time.

        :param ds: The dataset that contains only vectors (single axis tensors) with the elements
        that should be tokenized
        :return:
        """
        for features in ds:
            self.tokenize(features)

    def get_vocab(self):
        return self._vocab

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def enable_extensibility(self):
        """
        Switches extensibility to True and enables the dynamic tokenizing of new items

        :return: True
        """
        self._extensible = True
        return True

    def disable_extensibility(self):
        """
        Switches extensibility to False and disables the dynamic tokenizing of new items -> unknown items will
        throw an error from now on

        :return: True
        """
        self._extensible = False
        return True

    @property
    @abc.abstractmethod
    def identifier(self):
        pass

    @abc.abstractmethod
    def clear_vocab(self):
        pass

    @abc.abstractmethod
    def tokenize(self, input, progress_bar: bool = False):
        """
        Convert (virtually any) input to a tokenizer-specific token.
        To support multiple input types, this method may call specific private methods
        like `_tokenize_string(string)` to tokenize specific input.

        :param input: To be converted
        :param progress_bar: May enable tqdm progress bar
        :return: Token (may be other string or integer)
        """
        pass

    @abc.abstractmethod
    def detokenize(self, token, drop_tokens: list[str] = None, progress_bar: bool = False):
        """
        Return the corresponding string the given `token` represents

        :param token: Should be converted back to original value
        :param drop_tokens: A list of the string representation of the tokens that should be dropped on conversion
        :param progress_bar: May enable tqdm progress bar
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
        return self._max_len
