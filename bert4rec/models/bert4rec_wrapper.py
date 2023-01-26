from absl import logging
import json
import pathlib
import tensorflow as tf
import tensorflow_models as tfm
from typing import Union

from bert4rec.models.components import networks
from bert4rec.models.bert4rec_model import BERT4RecModel
from bert4rec.models.model_wrapper import ModelWrapper
import bert4rec.models.model_utils as utils
from bert4rec import tokenizers
from bert4rec.trainers import optimizers, trainer_utils

_ENCODER_CONFIG_FILE_NAME = "encoder_config.json"
_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"
_MODEL_WEIGHTS_FILES_PREFIX = "model_weights"


class BERT4RecModelWrapper(ModelWrapper):
    def __init__(self, bert_model: BERT4RecModel):
        super().__init__(bert_model)
        self.bert_model = bert_model
        self.update_meta(
            {
                "model": "BERT4Rec",
                "encoder_config": self.bert_model.encoder.get_config()
            }
        )

    def save(self, save_path: pathlib.Path, tokenizer: tokenizers.BaseTokenizer = None, mode: int = 0) -> True:
        """
        Saves a model to the disk. If the tokenizer is given, saves the used vocab from the tokenizer as well.

        NOTE: The model can only be saved successfully, when completely initialized. See model doc (further above)
        for more information.

        :param save_path: The path to save the model to
        :param tokenizer: If given, saves the used vocab from this tokenizer in the model directory
        :param mode: The mode determines where the model is stored. Available modes are 0, 1, 2.
        0 is default and stores the model relative to the project root. 1 stores the model relative
        to the virtual environment and 2 stores it relative to the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: True
        """
        save_path = utils.determine_model_path(save_path, mode)

        if self.bert_model.compiled_loss is None:
            raise RuntimeError("The model can't be saved without a loss. The model needs to be compiled first.")

        if self.bert_model.compiled_metrics and not self.bert_model.compiled_metrics._built:
            raise RuntimeError("The model can't be saved yet, as it is not fully instantiated and will "
                               "throw an error during saving. See model docs for more information.")

        logging.info(f"Saving {self.bert_model} to {save_path}")

        self.bert_model.save(save_path)

        if tokenizer:
            tokenizer.export_vocab_to_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
            self.update_meta({"tokenizer": tokenizer.identifier})

        # save meta config to file
        with open(save_path.joinpath(_META_CONFIG_FILE_NAME), "w") as f:
            json.dump(self._meta_config, f, indent=4)

        return True

    @classmethod
    def load(cls, save_path: pathlib.Path, mode: int = 0) -> dict:
        """
        Loads and returns all available assets in conjunction with the ml model. Loads at least a saved
        BERT4Rec model. Depending on the configuration might also load tokenizer with learnt vocab

        :param save_path: Path to the model directory
        :param mode: The mode determines from where the model is loaded. Available modes are 0, 1, 2.
        0 is default and loads the model relative from the project root. 1 loads the model relative
        from the virtual environment and 2 loads it relative from the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: Dict with all loaded assets
        """
        save_path = utils.determine_model_path(save_path, mode)

        if not save_path.exists():
            raise ValueError(f"The given path {save_path} does not exist.")

        logging.info(f"Loading model from {save_path}")

        loaded_assets = dict()
        loaded_bert = tf.keras.models.load_model(save_path, custom_objects={
            "BERT4RecModel": BERT4RecModel,
            "Bert4RecEncoder": networks.Bert4RecEncoder,
            "OnDeviceEmbedding": tfm.nlp.layers.OnDeviceEmbedding,
            "PositionEmbedding": tfm.nlp.layers.PositionEmbedding,
            "SelfAttentionMask": tfm.nlp.layers.SelfAttentionMask,
            "TransformerEncoderBlock": tfm.nlp.layers.TransformerEncoderBlock,
            "RelativePositionEmbedding": tfm.nlp.layers.RelativePositionEmbedding,
            "RelativePositionBias": tfm.nlp.layers.RelativePositionBias,
            "AdamWeightDecay": optimizers.AdamWeightDecay,
            "masked_accuracy": trainer_utils.masked_accuracy,
            "MaskedSparseCategoricalCrossentropy": trainer_utils.MaskedSparseCategoricalCrossentropy,
        })

        wrapper = cls(loaded_bert)
        loaded_assets["model_wrapper"] = wrapper

        try:
            with open(save_path.joinpath(_META_CONFIG_FILE_NAME)) as jf:
                meta_config = json.load(jf)
                wrapper._meta_config = meta_config

                if "tokenizer" in meta_config and meta_config["tokenizer"] is not None:
                    tokenizer = tokenizers.get(meta_config["tokenizer"])
                    tokenizer.import_vocab_from_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
                    loaded_assets["tokenizer"] = tokenizer

        except FileNotFoundError:
            logging.error(f"The meta configuration/information json file ({_META_CONFIG_FILE_NAME}) could "
                          f"not be found in the supposed model directory: {save_path}")

        return loaded_assets
