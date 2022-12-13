from absl import logging
import pathlib
import random

from bert4rec.apps import Recommender
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
    vocab_size = dataloader.tokenizer.get_vocab_size()
    special_tokens_length = len(dataloader._SPECIAL_TOKENS)

    recommender_app = Recommender(model, dataloader)

    # random movie list
    random_movie_indexes = [random.randint(special_tokens_length + 1, vocab_size) for _ in range(5)]
    movie_list_1 = dataloader.tokenizer.detokenize(random_movie_indexes)
    logging.info("First movie list:")
    logging.info(movie_list_1)
    recommendation_1 = recommender_app(movie_list_1)
    logging.info("Recommendation for first movie list:")
    logging.info(recommendation_1)

    # animated/children movie list
    movie_list_2 = [
        "Toy Story (1995)",
        "Toy Story 2 (1999)",
        "Pocahontas (1995)",
        "Lion King, The (1994)",
        "Aladdin (1992)",
        "Space Jam (1996)"
    ]
    logging.info("Second movie list:")
    logging.info(movie_list_2)
    recommendation_2 = recommender_app(movie_list_2)
    logging.info("Recommendation for second movie list:")
    logging.info(recommendation_2)

    # action/thriller movie list
    movie_list_3 = [
        "Ghost (1990)",
        "Purple Noon (1960)",
        "Maximum Risk (1996)",
        "Dante's Peak (1997)",
        "Mercury Rising (1998)",
        "Rope (1948)",
        "Siege, The (1998)",
        "Thirteenth Floor, The (1999)",
        "Someone to Watch Over Me (1987)",
        "Ninth Gate, The (2000)",
        "Firestarter (1984)"
    ]
    logging.info("Third movie list:")
    logging.info(movie_list_3)
    recommendation_3 = recommender_app(movie_list_3)
    logging.info("Recommendation for third movie list:")
    logging.info(recommendation_3)


if __name__ == "__main__":
    main()
