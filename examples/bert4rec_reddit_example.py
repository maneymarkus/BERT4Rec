from absl import logging
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.evaluation import BERT4RecEvaluator
from bert4rec.models.components import networks
from bert4rec.models import BERT4RecModel, BERT4RecModelWrapper, model_utils
from bert4rec import trainers
from bert4rec.trainers import optimizers
import bert4rec.utils as utils


def main():
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    EPOCHS = 200

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader_config = {
        "input_duplication_factor": 1
    }
    dataloader = dataloader_factory.create_reddit_dataloader(**dataloader_config)
    dataloader.generate_vocab()
    train_ds, val_ds, test_ds = dataloader.prepare_training()
    tokenizer = dataloader.get_tokenizer()

    # load a specific config
    config_path = pathlib.Path("../config/bert4rec_train_configs/reddit_128.json")
    config = utils.load_json_config(config_path)

    bert_encoder = networks.Bert4RecEncoder(tokenizer.get_vocab_size(), **config)
    model = BERT4RecModel(bert_encoder)
    model_wrapper = BERT4RecModelWrapper(model)
    # makes sure the weights are built
    _ = model(model.inputs)

    # set up trainer
    trainer = trainers.get(**{"model": model})
    trainer.initialize_model()

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_reddit"))
    # is needed as this does not create a new folder but rather the base name (or prefix)
    # for the created checkpoint files
    checkpoint_path = save_path.joinpath("checkpoints")

    train_batches = dataloader_utils.make_batches(train_ds, batch_size=64)
    val_batches = dataloader_utils.make_batches(val_ds, batch_size=64)
    test_batches = dataloader_utils.make_batches(test_ds, batch_size=64)

    # train the model
    trainer.train(train_batches,
                  val_batches,
                  checkpoint_path=checkpoint_path,
                  epochs=EPOCHS,
                  steps_per_epoch=500,
                  validation_steps=10000)
    trainer.update_wrapper_meta_info(model_wrapper, dataloader)

    evaluator = BERT4RecEvaluator(dataloader=dataloader)

    metrics_objects = evaluator.evaluate(model, test_batches)
    evaluator.save_results(save_path)
    metrics = evaluator.get_metrics_results()
    print(metrics)

    model_wrapper.save(save_path=save_path, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
