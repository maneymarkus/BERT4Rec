from absl import logging
import os
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.model.components import networks
from bert4rec.model import BERTModel
import bert4rec.utils as utils


def main():
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader_config = {
        "input_duplication_factor": 1
    }
    dataloader = dataloader_factory.create_ml_1m_dataloader(**dataloader_config)
    dataloader.generate_vocab()
    train_ds, val_ds, test_ds = dataloader.prepare_training()
    tokenizer = dataloader.get_tokenizer()

    # load a specific config
    config_path = pathlib.Path("../config/bert_train_configs/ml-1m_128.json")
    config = utils.load_json_config(config_path)

    bert_encoder = networks.BertEncoder(tokenizer.get_vocab_size(), **config)
    model = BERTModel(bert_encoder)

    train_batches = dataloader_utils.make_batches(train_ds)
    for b in train_batches.take(1):
        example = b

    ####################################
    # without using masked lm layer    #
    ####################################
    embedding_table = model.encoder.get_embedding_table()
    logging.info("Embedding Table:")
    # shape should be (vocab size, embedding width)
    logging.info(embedding_table)
    predictions = model(example)
    y_true = example["masked_lm_ids"]
    sequence_output = predictions["sequence_output"]
    masked_token_sequence = tf.gather(sequence_output, example["masked_lm_positions"], axis=1, batch_dims=1)
    y_pred = tf.linalg.matmul(masked_token_sequence, embedding_table, transpose_b=True)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
    loss = scce(y_true, y_pred)
    logging.info("y_true:")
    logging.info(y_true)
    logging.info(y_true.shape)
    logging.info("y_pred:")
    logging.info(y_pred)
    logging.info(y_pred.shape)
    logging.info("loss:")
    logging.info(loss)

    ####################################
    # with using masked lm layer       #
    ####################################
    logging.info("y_true")
    logging.info(y_true)
    y_pred_mlm = predictions["mlm_logits"]
    logging.info("y_pred_mlm:")
    logging.info(y_pred_mlm)
    # shape should be (num batches, num masked tokens/num predictions, vocab size)
    logging.info(y_pred_mlm.shape)
    loss = scce(y_true, y_pred_mlm)
    logging.info("mlm_logits_loss:")
    logging.info(loss)


if __name__ == "__main__":
    main()
