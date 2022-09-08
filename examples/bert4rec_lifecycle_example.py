from absl import logging
import os
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.model.components import networks
from bert4rec.model import BERTModel, BERT4RecModelWrapper, model_utils
from bert4rec.trainers import BERT4RecTrainer


def main():
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader = dataloader_factory.create_ml_1m_dataloader()
    dataloader.generate_vocab()
    ds = dataloader.preprocess_dataset()
    tokenizer = dataloader.get_tokenizer()

    bert_encoder = networks.BertEncoder(tokenizer.get_vocab_size())
    model = BERTModel(bert_encoder)
    model_wrapper = BERT4RecModelWrapper(model)
    # makes sure the weights are built
    _ = model(model.inputs)
    embedding_table = model.encoder._embedding_layer.embeddings
    #print(embedding_table.shape)

    trainer = BERT4RecTrainer(model)
    trainer.initialize_model()

    partial_ds = ds.take(100)
    batched_partial_ds = dataloader_utils.make_batches(partial_ds, buffer_size=100)
    for b in batched_partial_ds.take(1):
        example = b

    predictions = model(example)
    y_true = example["masked_lm_ids"]
    sequence_output = predictions["sequence_output"]
    masked_token_sequence = tf.gather(sequence_output, example["masked_lm_positions"], axis=1, batch_dims=1)
    y_pred = tf.linalg.matmul(masked_token_sequence, embedding_table, transpose_b=True)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = scce(y_true, y_pred)
    print("y_true:")
    print(y_true)
    print(y_true.shape)
    print("y_pred:")
    print(y_pred)
    print(y_pred.shape)
    print("loss:")
    print(loss)

    save_path = model_utils.determine_model_path(pathlib.Path("my_model"))
    checkpoint_path = save_path.joinpath("checkpoints")

    # train_ds, val_ds, test_ds = dataloader_utils.split_dataset(ds, shuffle_size=2000)
    train_ds, val_ds, test_ds = dataloader_utils.split_dataset(ds, shuffle_size=100)
    # train_batches = dataloader_utils.make_batches(train_ds, batch_size=16)
    train_batches = dataloader_utils.make_batches(train_ds, batch_size=16, buffer_size=100)
    # val_batches = dataloader_utils.make_batches(val_ds, batch_size=16)
    val_batches = dataloader_utils.make_batches(val_ds, batch_size=16, buffer_size=100)
    # test_batches = dataloader_utils.make_batches(test_ds, batch_size=16)
    test_batches = dataloader_utils.make_batches(test_ds, batch_size=16, buffer_size=100)

    trainer.train(train_batches, val_batches, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
