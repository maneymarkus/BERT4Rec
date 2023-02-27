from absl import logging
import random
import string

from bert4rec import datasets, dataloaders
from bert4rec.dataloaders import preprocessors


if __name__ == "__main__":
    # Exemplary usage of the ML_1M Dataloader for the BERT4Rec Model
    # FIRST: use default values
    logging.set_verbosity(logging.DEBUG)
    dataloader = dataloaders.BERT4RecML1MDataloader()
    dataloader.generate_vocab()
    tokenizer = dataloader.get_tokenizer()
    train_ds, val_ds, test_ds = dataloader.prepare_training()
    test_data = [random.choice(string.ascii_letters) for _ in range(25)]
    #logging.debug(test_data)
    model_input = dataloader.prepare_inference(test_data)
    #logging.debug(model_input)
    tensor = model_input["input_word_ids"]
    detokenized = tokenizer.detokenize(tensor, [dataloader._PAD_TOKEN])
    batched_ds = dataloaders.dataloader_utils.make_batches(train_ds, buffer_size=100)
    for b in batched_ds.take(1):
        print(b)

    # SECOND: Use another preprocessor and extract more data
    dataloader_2 = dataloaders.BERT4RecML1MDataloader(preprocessor=preprocessors.BERT4RecTemporalPreprocessor)
    dataloader_2.generate_vocab()
    tokenizer = dataloader_2.get_tokenizer()
    train_ds, val_ds, test_ds = dataloader_2.prepare_training(extract_data=["movie_name", "timestamp"],
                                                              datatypes=["list", "list"])
    test_data_sequence = [random.choice(string.ascii_letters) for _ in range(25)]
    # step is 8000 to generate 25 numbers (to be compatible with the 25 random values generated above
    test_data_timestamps = list(range(10000000, 10200000, 8000))
    test_data = (test_data_sequence, test_data_timestamps)
    # logging.debug(test_data)
    model_input = dataloader_2.prepare_inference(test_data)
    #logging.debug(model_input)
    tensor = model_input["input_word_ids"]
    detokenized = tokenizer.detokenize(tensor, [dataloader_2._PAD_TOKEN])
    batched_ds = dataloaders.dataloader_utils.make_batches(train_ds, buffer_size=100)
    for b in batched_ds.take(1):
        print(b)

    # Additional Info: only work with toy dataset
    datalaoder_3 = dataloader = dataloaders.BERT4RecML1MDataloader(data_source=datasets.ML1M.set_load_n_records(50000))
    # omitted as the rest works just as usual
