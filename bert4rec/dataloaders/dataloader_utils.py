"""
This file contains basic utility function concerning dataloaders, like e.g. content trimming and padding
"""

from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text


def convert_df_to_ds(df: pd.DataFrame, datatypes: list[str] = None):
    """
    Converts a given dataframe 'linearly' (column-by-column) into a dataset and returns it.
    Datatypes of the columns may be stated to improve conversion results (in case of e.g. 'lists')

    :param df: The dataframe to be converted
    :param datatypes: A list (as long as the number of available columns in the dataframe) of strings
    describing each datatype of the content of the columns. May either be 'list' or numpy datatypes (e.g. int64)
    :return: A tf.data.Dataset object
    """
    c_index = 0

    # get first column extra to instantiate dataset
    datatype = None
    if datatypes is not None:
        datatype = datatypes[0]
    ds = convert_column_to_ds(df.iloc[:, 0], datatype)

    for _, column in df.iteritems():
        if c_index == 0:
            c_index += 1
            continue

        if datatypes is not None:
            datatype = datatypes[c_index]
        part_ds = convert_column_to_ds(column, datatype)

        ds = tf.data.Dataset.zip((ds, part_ds))
        c_index += 1
    return ds


def convert_column_to_ds(column: pd.Series, datatype: str = None):
    """
    Converts a single column to a `tf.data.Dataset` object. The datatype may define the data type of the content
    of the column.

    :param column: The column to be converted
    :param datatype: A string describing each datatype of the content of the columns.
    May either be 'list' or numpy datatypes (e.g. int64)
    :return: A tf.data.Dataset object
    """
    if datatype is not None:
        if datatype == "list":
            ds = tf.data.Dataset.from_tensor_slices(
                tf.ragged.constant(column.to_numpy())
            )
        else:
            ds = tf.data.Dataset.from_tensor_slices(
                column.to_numpy().astype(datatype)
            )
    else:
        logging.info(f"No datatypes were given, "
                     f"so the datatypes of the columns are tried to be inferred by tensorflow.")
        ds = tf.data.Dataset.from_tensor_slices(
            column.to_numpy()
        )
    return ds


def trim_content(
        max_seq_length: int,
        content: list, trimmer=None) \
        -> tuple[tf_text.RoundRobinTrimmer, list]:
    """
    Cut off `content` at given `max_seq_length` according to used `trimmer`. Unfortunately, the base trimmer
    class is not referable and can't therefore be used to typehint the trimmer.

    :param trimmer: A trimmer from tensorflow_text inheriting from tensorflow_text.Trimmer
    :param max_seq_length:
    :param content:
    :return: trimmer, trimmed content
    """
    if trimmer is not None:
        if not hasattr(trimmer, "trim") or not callable(getattr(trimmer, "trim")):
            raise ValueError("The given trimmer might not be of type tensorflow_text.trimmer, "
                             "as it does not have the required method (trim()).")

    if trimmer is None:
        trimmer = tf_text.RoundRobinTrimmer(max_seq_length=max_seq_length)

    trimmed_segments = trimmer.trim(content)
    return trimmer, trimmed_segments


def apply_dynamic_masking_task(segments: tf.RaggedTensor,
                               max_selections_per_batch: int,
                               mask_token_id: int,
                               special_token_ids: list[int],
                               vocab_size: int,
                               selection_rate: float = 0.2,
                               mask_token_rate: float = 0.8,
                               random_token_rate: float = 0.1) \
        -> tuple[
            tf_text.RandomItemSelector, tf_text.MaskValuesChooser, tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """
    Applies dynamic masking task as described in https://arxiv.org/abs/1810.04805
    with the help of tf_text.RandomItemSelector and tf_text.MaskValuesChooser

    :param segments: tf.RaggedTensor containing the segments to be masked
    :param max_selections_per_batch: Maximum amount of selections per batch to be masked
    :param special_token_ids: Special tokens that shouldn't be selected as a masking target
    and shouldn't be inserted as random tokens
    :param mask_token_id:
    :param vocab_size:
    :param selection_rate: Percentage of segments that should be selected for the masking task
    :param mask_token_rate: Percentage of selected segments that should be masked with the masking token
    :param random_token_rate: Percentage of selected segments that should be switched with a random token
    :return:
    """
    random_selector = tf_text.RandomItemSelector(max_selections_per_batch=max_selections_per_batch,
                                                 selection_rate=selection_rate,
                                                 unselectable_ids=special_token_ids)
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size=vocab_size,
                                                    mask_token=mask_token_id,
                                                    mask_token_rate=mask_token_rate,
                                                    random_token_rate=random_token_rate)
    masked_token_ids, masked_lm_positions, masked_lm_ids = tf_text.mask_language_model(
        segments,
        random_selector,
        mask_values_chooser
    )
    return random_selector, mask_values_chooser, masked_token_ids, masked_lm_positions, masked_lm_ids


def split_dataset(ds: tf.data.Dataset,
                  ds_size: int = None,
                  train_split: float = 0.8,
                  val_split: float = 0.1,
                  test_split: float = 0.1,
                  shuffle: bool = True,
                  shuffle_size: int = 10000) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if (train_split + test_split + val_split) != 1:
        raise ValueError("The dataset can only be split in parts that sum up to 1 or a 100%.")

    if ds_size is None:
        ds_size = tf.data.experimental.cardinality(ds)
        if ds_size == tf.data.experimental.UNKNOWN_CARDINALITY:
            raise ValueError("Unfortunately, the size of the dataset couldn't be determined."
                             "You could try to set the dataset size as a parameter when calling this function."
                             "Note: splitting dynamic datasets is not implemented/supported yet.")
        # convert determined dataset size from scalar tensor to simple integer value
        ds_size = ds_size.numpy()

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def make_batches(dataset: tf.data.Dataset,
                 buffer_size: int = 2000,
                 batch_size: int = 64,
                 squeeze_tensors: bool = True):
    """
    Combines consecutive elements of the given dataset into batches. Tensors may be squeezed if wanted,
    to prevent elements of the batched dataset to have a shape of [batch_size, 1, tokens].

    :param dataset: The dataset that should be batched
    :param buffer_size: The buffer size for the shuffle.
    See: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    :param batch_size: The size of the batches to be generated.
    See: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
    :param squeeze_tensors: Determines if the tensor elements of this dataset should be squeezed (reduce
    the dimensions). Tensors can only be squeezed, if they have an "empty" dimension (e.g. a shape
    like this: [x, 1, y] has a middle empty dimension that can possibly be removed). Empty dimension might
    occur due to preprocessing.
    :return:
    """
    def squeeze_func(d: dict):
        for key, tensor in d.items():
            d[key] = tf.squeeze(tensor)
        return d

    return dataset \
        .shuffle(buffer_size) \
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda d: squeeze_func(d) if squeeze_tensors else d) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)
