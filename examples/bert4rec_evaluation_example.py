from absl import logging
import pathlib

from bert4rec.dataloaders import get_dataloader_factory, dataloader_utils, samplers
from bert4rec.evaluation import BERT4RecEvaluator
from bert4rec.models import BERT4RecModelWrapper, model_utils


def main():
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # set logging to most verbose level
    logging.set_verbosity(logging.DEBUG)

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_ml-1m_15"))

    loaded_assets = BERT4RecModelWrapper.load(save_path)
    loaded_wrapper = loaded_assets["model_wrapper"]
    loaded_model = loaded_wrapper.model
    dataloader_config = {}
    if "tokenizer" in loaded_assets:
        tokenizer = loaded_assets["tokenizer"]
        dataloader_config["tokenizer"] = tokenizer

    dataloader_factory = get_dataloader_factory("bert4rec")
    dataloader = dataloader_factory.create_ml_1m_dataloader(**dataloader_config)
    dataloader.generate_vocab()
    train_ds, val_ds, test_ds = dataloader.prepare_training()

    test_batches = dataloader_utils.make_batches(test_ds, batch_size=256)

    sampler_config = {
        "sample_size": 100,
        "allow_duplicates": False
    }
    sampler = samplers.get("random", **sampler_config)
    evaluator = BERT4RecEvaluator(dataloader=dataloader)

    metrics_objects = evaluator.evaluate(loaded_model, test_batches)
    #evaluator.save_results(save_path)
    metrics = evaluator.get_metrics_results()
    print(metrics)


if __name__ == "__main__":
    main()
