from absl import logging
import pathlib
import tensorflow as tf

from bert4rec.apps import Ranker
from bert4rec import dataloaders
from bert4rec.model import BERT4RecModelWrapper, model_utils


def main():
    logging.set_verbosity(logging.INFO)

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_ml-1m_with_adamw"))

    loaded_assets = BERT4RecModelWrapper.load(save_path)
    loaded_wrapper = loaded_assets["model_wrapper"]
    model = loaded_wrapper.bert_model
    dataloader_config = {}
    if "tokenizer" in loaded_assets:
        tokenizer = loaded_assets["tokenizer"]
        dataloader_config["tokenizer"] = tokenizer

    dataloader_factory = dataloaders.get_dataloader_factory()
    dataloader = dataloader_factory.create_ml_1m_dataloader(**dataloader_config)
    dataloader.generate_vocab()

    ds = dataloader.load_data_into_ds()
    for e in ds.shuffle(ds.cardinality()).take(1):
        # dataset is tuple (user id, sequence) and usage of decode to get str array instead of byte array
        example = [x.decode() for x in e[1].numpy().tolist()]

    ground_truth = example[-1]
    example = example[:-1]
    logging.info(example)
    logging.info(ground_truth)

    popular_items = dataloader.create_popular_item_ranking()[:100]

    if ground_truth not in popular_items:
        popular_items.append(ground_truth)

    ranker_app = Ranker(model, dataloader)

    rank, assert_string = ranker_app(example, ground_truth, popular_items)
    #rank, assert_string = ranker_app(example, ground_truth)

    print(assert_string)


if __name__ == "__main__":
    main()