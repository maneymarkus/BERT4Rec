from absl import logging
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.evaluation import BERT4RecEvaluator
from bert4rec.models.components import networks
from bert4rec.models import BERT4RecModel, BERT4RecModelWrapper, model_utils
import bert4rec.trainers as trainers
import bert4rec.utils as utils


def main():
    #os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader = dataloader_factory.create_ml_1m_dataloader()
    dataloader.generate_vocab()
    train_ds, val_ds, test_ds = dataloader.prepare_training()
    tokenizer = dataloader.get_tokenizer()

    # load a specific config
    config_path = pathlib.Path("../config/bert4rec_train_configs/ml-1m_128.json")
    config = utils.load_json_config(config_path)

    bert_encoder = networks.Bert4RecEncoder(tokenizer.get_vocab_size(), **config)
    model = BERT4RecModel(bert_encoder)
    model_wrapper = BERT4RecModelWrapper(model)
    # makes sure the weights are built
    _ = model(model.inputs)

    # set up trainer to initialize model
    trainer = trainers.get(**{"model": model})
    trainer.initialize_model()

    train_batches = dataloader_utils.make_batches(train_ds.take(64))
    test_batches = dataloader_utils.make_batches(test_ds.take(64))

    # do short training to build compiled metrics (necessary for saving)
    model.fit(train_batches)

    evaluator = BERT4RecEvaluator(dataloader=dataloader)

    evaluator.evaluate(model_wrapper, test_batches)
    print(evaluator.get_metrics_results())

    save_path = pathlib.Path("example_save_model")
    model_wrapper.save(save_path)

    reloaded_assets = BERT4RecModelWrapper.load(save_path)
    reloaded_model_wrapper = reloaded_assets["model_wrapper"]

    evaluator.reset_metrics()

    evaluator.evaluate(reloaded_model_wrapper, test_batches)
    print(evaluator.get_metrics_results())


if __name__ == "__main__":
    main()
