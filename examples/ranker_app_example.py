from absl import logging
import pathlib

from bert4rec.apps import Ranker
from bert4rec import dataloaders
from bert4rec.models import BERT4RecModelWrapper, model_utils


def main():
    logging.set_verbosity(logging.INFO)

    save_path = model_utils.determine_model_path(pathlib.Path("bert4rec_ml-1m_3"))

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

    ranker_app = Ranker(model, dataloader)

    ds = dataloader.load_data(split_data=False)[0]
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

    rank_1, assert_string_1 = ranker_app(example, ground_truth, popular_items)
    rank_2, assert_string_2 = ranker_app(example, ground_truth)

    print(assert_string_1)
    print(assert_string_2)


if __name__ == "__main__":
    main()
