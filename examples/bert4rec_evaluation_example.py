from absl import logging
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils
from bert4rec.evaluation import BERT4RecEvaluator
from bert4rec.models.components import networks
from bert4rec.models import BERTModel, BERT4RecModelWrapper, model_utils
from bert4rec import trainers
from bert4rec.trainers import optimizers
import bert4rec.utils as utils


def main():
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_ml-1m_with_adamw"))

    loaded_assets = BERT4RecModelWrapper.load(save_path)
    loaded_wrapper = loaded_assets["model_wrapper"]
    model = loaded_wrapper.bert_model
    dataloader_config = {}
    if "tokenizer" in loaded_assets:
        tokenizer = loaded_assets["tokenizer"]
        dataloader_config["tokenizer"] = tokenizer

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader = dataloader_factory.create_ml_1m_dataloader(**dataloader_config)
    dataloader.generate_vocab()
    train_ds, val_ds, test_ds = dataloader.prepare_training()

    test_batches = dataloader_utils.make_batches(test_ds)

    evaluator = BERT4RecEvaluator()

    metrics_objects = evaluator.evaluate(loaded_wrapper, test_batches, dataloader)
    #evaluator.save_results(save_path)
    metrics = evaluator.get_metrics_results()
    print(metrics)


if __name__ == "__main__":
    main()
