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

    if datatypes is not None:
        if len(datatypes) != len(df.columns):
            raise ValueError(f"The given datatypes list ({datatypes}, len: {len(datatypes)}) has "
                             f"to have as many elements as columns in the given df "
                             f"({len(df.columns)}).")

    datatype = None

    logging.info("Start to convert dataframe to dataset")
    datasets = tuple()
    for i, column_tuple in enumerate(df.items()):
        column = column_tuple[1]

        if datatypes is not None:
            datatype = datatypes[i]
        part_ds = convert_column_to_ds(column, datatype)

        datasets += (part_ds,)
    ds = tf.data.Dataset.zip(datasets)
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


def make_sequence_df(df: pd.DataFrame,
                     group_column_name: str,
                     extract_sequences: list,
                     min_sequence_length: int = 0) -> pd.DataFrame:
    """
    Transforms a given pd.DataFrame into a sequence based dataset and extracts the wanted sequence
    information by column name(s)

    :param df:
    :param group_column_name: The column name by which the pd.DataFrame should be grouped
    :param extract_sequences: The sequences that should be extracted (by column name)
    :param min_sequence_length: Determines the minimum length for sequences to be extracted
    """
    df = df.groupby(group_column_name)
    sequence_df = pd.DataFrame(columns=extract_sequences)
    for group, group_data in df:
        seq_data = {}
        for seq_name in extract_sequences:
            sequence = group_data[seq_name].to_list()
            # only keep sequences with at least `min_sequence_length` items
            if len(sequence) < min_sequence_length:
                break

            seq_data[seq_name] = sequence

        # add data dict as a single row to a new pd.DataFrame
        single_seq_df = pd.DataFrame([seq_data])
        sequence_df = pd.concat([sequence_df, single_seq_df], ignore_index=True)
    return sequence_df


def split_sequence_df(
        df: pd.DataFrame,
        group_by_column: str,
        extract_columns: list,
        min_sequence_length: int = 5
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits a given dataframe with sequence data in three dataframes (for training, validation and testing).
    The dataframe is grouped by the given `group_by_column` and all the given `extract_columns`
    will be extracted and part of the new dataframe. The split is always done the same: The first
    n-2 elements will be used for training, the first n-1 elements will be used for validation
    and the whole sequence will be used for testing

    :param df: pd.DataFrame
    :param group_by_column: The key of the column to group by
    :param extract_columns: The keys of the columns that should be extracted
    :param min_sequence_length: Only keep sequences with at least `min_sequence_length` items
    :return:
    """
    if group_by_column not in df.columns:
        raise ValueError(f"Group column key {group_by_column} is not present in columns "
                         f"in dataframe: {df.columns}")

    if len(extract_columns) - 1 > len(df.columns):
        raise ValueError(f"More columns to extract have been given than there are actual columns "
                         f"in the dataframe: {len(df.columns)}")

    for col in extract_columns:
        if col not in df.columns:
            raise ValueError(f"Column key {col} of the extract_columns argument is not present "
                             f"in columns in dataframe: {df.columns}")

    grouped_df = df.groupby(group_by_column)
    train_ds = {}
    val_ds = {}
    test_ds = {}

    # iterate over groups in grouped_df
    logging.info("Split dataframe:")
    for i, group_data in enumerate(tqdm.tqdm(grouped_df)):
        group, group_content = group_data
        train_ds[i] = {}
        val_ds[i] = {}
        test_ds[i] = {}
        for col in extract_columns:
            sequence = group_content[col].to_list()

            # take all elements for train before checking if there are enough elements to split this sequence
            train_ds[i][col] = sequence
            if len(sequence) >= min_sequence_length:
                # for train take the first n-2 elements
                train_ds[i][col] = sequence[:-2]
                # for validation take the first n-1 elements
                val_ds[i][col] = sequence[:-1]
                # for testing take all the elements
                test_ds[i][col] = sequence

    train_df = pd.DataFrame.from_dict(train_ds, orient="index")
    val_df = pd.DataFrame.from_dict(val_ds, orient="index")
    test_df = pd.DataFrame.from_dict(test_ds, orient="index")

    return train_df, val_df, test_df


def duplicate_dataset(ds: tf.data.Dataset, duplication_factor: int) -> tf.data.Dataset:
    if duplication_factor < 1:
        raise ValueError(f"A duplication factor of less than 1 (given: {duplication_factor}) "
                         "is not allowed!")
    if duplication_factor > 1:
        ds = ds.repeat(duplication_factor)
    return ds


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
    # select num_to_predict indexes from pos_indexes and sort again
    pos_indexes = pos_indexes[:num_to_predict]
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


def mask_last_token_only(tensor: tf.Tensor, mask_token_id: int):
    tensor_values = tensor.numpy()
    masked_lm_ids = tf.constant([tensor_values[-1]], dtype=tf.int64)
    tensor_values[-1] = mask_token_id
    masked_token_ids = tf.constant(tensor_values, dtype=tf.int64)
    masked_lm_positions = tf.constant([(len(tensor_values) - 1)], dtype=tf.int64)
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
                 reshuffle_each_iteration: bool = False,
                 seed: int = None) -> tf.data.Dataset:
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
        .shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=seed) \
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda d: squeeze_func(d) if squeeze_tensors else d) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)
