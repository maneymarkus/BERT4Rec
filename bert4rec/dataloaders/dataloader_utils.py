"""
This file contains basic utility function concerning dataloaders, like e.g. content trimming and padding
"""

from absl import logging
import collections
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tqdm


def rank_items_by_popularity(items: list) -> list:
    sorted_item_list = sorted(items, key=collections.Counter(items).get, reverse=True)
    # remove duplicates
    sorted_items_list = list(dict.fromkeys(sorted_item_list))
    return sorted_items_list


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

    logging.info("Start to convert dataframe to dataset")
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


def split_sequence_df(
        df: pd.DataFrame,
        group_by_column: str,
        sequence_column: str,
        min_sequence_length: int = 5
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits a given dataframe with sequence data in three dataframes (for training, validation and testing).
    It is assumed that the `group_by_column` column contains the user (or subject) and the `sequence_column`
    contains the item sequence. The split is always done the same: The first n-2 elements will be used
    for training, the first n-1 elements will be used for validation and the whole sequence will
    be used for testing

    :param df: A pd.Dataframe object with
    :param group_by_column: The key of the column to group by
    :param sequence_column: The key of the column that will contain the sequence data of interest
    :param min_sequence_length: Only keep sequences with at least `min_sequence_length` items
    :return:
    """
    if group_by_column not in df.columns:
        raise ValueError(f"Group column key {group_by_column} is not present in columns "
                         f"in dataframe: {df.columns}")

    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column key {sequence_column} is not present in columns "
                         f"in dataframe: {df.columns}")

    sequence_df_columns = [group_by_column, sequence_column]
    grouped_df = df.groupby(group_by_column)
    train_ds = {}
    val_ds = {}
    test_ds = {}

    # iterate over groups in grouped_df
    logging.info("Split dataframe:")
    for group, group_data in tqdm.tqdm(grouped_df):
        sequence = group_data[sequence_column].to_list()

        # take all elements for train before checking if there are enough elements to split this sequence
        train_ds[group] = sequence
        if len(sequence) >= min_sequence_length:
            # for train take the first n-2 elements
            train_ds[group] = sequence[:-2]
            # for validation take the first n-1 elements
            val_ds[group] = sequence[:-1]
            # for testing take all the elements
            test_ds[group] = sequence

    train_df = pd.DataFrame(list(train_ds.items()), columns=sequence_df_columns)
    val_df = pd.DataFrame(list(val_ds.items()), columns=sequence_df_columns)
    test_df = pd.DataFrame(list(test_ds.items()), columns=sequence_df_columns)

    return train_df, val_df, test_df


def split_df_into_three_ds(df: pd.DataFrame,
                           train_duplication_factor: int,
                           group_by_column: str,
                           sequence_column: str) \
        -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    train_df, val_df, test_df = split_sequence_df(df, group_by_column, sequence_column)
    datatypes = ["int64", "list"]
    train_ds = convert_df_to_ds(train_df, datatypes)
    val_ds = convert_df_to_ds(val_df, datatypes)
    test_ds = convert_df_to_ds(test_df, datatypes)

    if train_duplication_factor > 1:
        train_ds = train_ds.repeat(train_duplication_factor)

    return train_ds, val_ds, test_ds


def apply_dynamic_masking_task(sequence_tensor: tf.Tensor,
                               max_selections_per_seq: int,
                               mask_token_id: int,
                               special_token_ids: list[int],
                               vocab_size: int,
                               selection_rate: float = 0.2,
                               mask_token_rate: float = 0.8,
                               random_token_rate: float = 0.1,
                               seed: int = None) \
        -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Applies dynamic masking task as described in https://arxiv.org/abs/1810.04805

    :param sequence_tensor: one dimensional tensor containing the (already tokenized) sequence
        on which the mlm should be applied on
    :param max_selections_per_seq: Maximum amount of selections per batch to be masked
    :param special_token_ids: Special tokens that shouldn't be selected as a masking target
        and shouldn't be inserted as random tokens
    :param mask_token_id:
    :param vocab_size:
    :param selection_rate: Percentage of segments that should be selected for the masking task
    :param mask_token_rate: Percentage of selected segments that should be masked with the masking token
    :param random_token_rate: Percentage of selected segments that should be switched with a random token
    :param seed: The seed for random operations
    :return: tensor with the mlm applied
    """
    sequence = sequence_tensor.numpy()
    dtype = sequence_tensor.dtype

    # set random seed for reproducibility (default is None -> different seed each call)
    random.seed(seed)

    # remove special tokens from sequence to calculate how many predictions can be inserted by applying the mlm
    special_token_indexes = np.argwhere(np.isin(sequence, special_token_ids))
    sequence_without_special_tokens = np.delete(sequence, special_token_indexes)

    num_to_predict = min(
        max_selections_per_seq,
        max(
            1, int(len(sequence_without_special_tokens) * selection_rate)
        )
    )

    # generate selectable vocab for inserting random tokens in masked_lm
    selectable_vocab = [i for i in range(vocab_size) if i not in special_token_ids]

    # create a list of indexes (item positions) in the sequence that can possibly be masked
    pos_indexes = [i for i, _ in enumerate(sequence_without_special_tokens)]
    random.shuffle(pos_indexes)
    # only select num_to_predict - 1 indexes from pos_indexes and sort again
    pos_indexes = pos_indexes[:num_to_predict-1]
    # always mask the last element (train to predict)
    pos_indexes.append(len(sequence) - 1)
    pos_indexes.sort()

    masked_lm_ids = []
    masked_lm_positions = []
    masked_token_ids = sequence.copy()

    for index in pos_indexes:
        if len(masked_lm_ids) >= num_to_predict:
            break

        # keep original token in 1 - mask_token_rate + random_token_rate of the cases
        replaced_token = sequence[index]
        # masked language models
        rn = random.random()
        # insert random token at random_token_rate
        if rn < mask_token_rate + random_token_rate:
            replaced_token = random.choice(selectable_vocab)
        # insert masked token at mask_token_rate
        if rn < mask_token_rate:
            replaced_token = mask_token_id

        masked_token_ids[index] = replaced_token
        masked_lm_ids.append(sequence[index])
        masked_lm_positions.append(index)

    masked_token_ids = tf.constant(masked_token_ids, dtype=dtype)
    masked_lm_ids = tf.constant(masked_lm_ids, dtype=dtype)
    masked_lm_positions = tf.constant(masked_lm_positions, dtype=dtype)

    return masked_token_ids, masked_lm_positions, masked_lm_ids


def split_dataset(ds: tf.data.Dataset,
                  ds_size: int = None,
                  train_split: float = 0.8,
                  val_split: float = 0.1,
                  test_split: float = 0.1,
                  shuffle: bool = True,
                  shuffle_size: int = 10000,
                  seed: int = 12) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
        ds = ds.shuffle(shuffle_size, seed=seed)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def make_batches(dataset: tf.data.Dataset,
                 buffer_size: int = None,
                 batch_size: int = 64,
                 squeeze_tensors: bool = False,
                 reshuffle_each_iteration: bool = False) -> tf.data.Dataset:
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
    like this: [x, 1, y] has an empty middle dimension that can possibly be removed). Empty dimensions might
    occur due to preprocessing.
    :param reshuffle_each_iteration: See: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    :return:
    """
    def squeeze_func(d: dict):
        for key, tensor in d.items():
            d[key] = tf.squeeze(tensor)
        return d

    if buffer_size is None:
        buffer_size = tf.data.experimental.cardinality(dataset)

    if buffer_size == tf.data.experimental.UNKNOWN_CARDINALITY:
        raise ValueError(f"Since the buffer size was not given it was tried to determine the cardinality (size) "
                         f"of the dataset to use it as the buffer size. However, the size could not be "
                         f"determined. Please provide a buffer size.")

    return dataset \
        .shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration) \
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda d: squeeze_func(d) if squeeze_tensors else d) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)
