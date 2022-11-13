from absl import logging
import os
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.evaluation import BERT4RecEvaluator
from bert4rec.model.components import networks
from bert4rec.model import BERTModel, BERT4RecModelWrapper, model_utils
from bert4rec import trainers
from bert4rec.trainers import optimizers
import bert4rec.utils as utils


def main():
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader_config = {
        "input_duplication_factor": 10
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
    model_wrapper = BERT4RecModelWrapper(model)
    # makes sure the weights are built
    _ = model(model.inputs)

    # custom optimizer
    optimizer = optimizers.get("adamw", **{"num_train_steps": 400000})

    # set up trainer
    trainer = trainers.get(**{"model": model})
    trainer.initialize_model(optimizer=optimizer)

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_ml-1m_with_adamw"))
    # is needed as this does not create a new folder but rather the base name for the created files
    checkpoint_path = save_path.joinpath("checkpoints")

    train_batches = dataloader_utils.make_batches(train_ds, batch_size=256)
    val_batches = dataloader_utils.make_batches(val_ds, batch_size=256)
    test_batches = dataloader_utils.make_batches(test_ds, batch_size=256)

    # TODO: save and load dataset to save time using the official tf.data.Dataset.save() and .load() api
    # not urgent anymore as tensors tend to have a much faster computation time (especially concerning
    # shuffle buffer filling) than ragged tensors
    # See: https://stackoverflow.com/a/67781967

    # set up a training loop callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_accuracy",
        patience=30,
        verbose=1,
        restore_best_weights=True
    )
    trainer.append_callback(early_stopping_callback)

    # train the model
    trainer.train(train_batches, val_batches, checkpoint_path=checkpoint_path, epochs=200)

    evaluator = BERT4RecEvaluator()

    metrics = evaluator.evaluate(model_wrapper, test_batches, dataloader)
    evaluator.save_results(save_path)
    print(metrics)


if __name__ == "__main__":
    main()
