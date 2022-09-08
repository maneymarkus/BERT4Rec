from absl import logging
import collections
import copy
import json
import pathlib
import random
import string
import tensorflow as tf
from typing import Union

from bert4rec.dataloaders.bert4rec_dataloaders import BERT4RecDataloader, BERT4RecML1MDataloader
import bert4rec.dataloaders.dataloader_utils as dataloader_utils
from bert4rec.model.components import layers, networks
import bert4rec.model.model_utils as utils
from bert4rec.tokenizers import BaseTokenizer, tokenizer_factory
from bert4rec.trainers import optimizers

_ENCODER_CONFIG_FILE_NAME = "encoder_config.json"
_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"
_MODEL_WEIGHTS_FILES_PREFIX = "model_weights"


train_step_signature = [{
    "labels": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_word_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_type_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_positions": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_weights": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
}]


class BERTModel(tf.keras.Model):
    """
    TODO: implement correct reloading
    NOTE: Reloading the model does not work properly, even with the custom_objects property set and all
    custom layers registered, `tf.keras.models.load_model()` does not reload into this class but into a
    tf.keras.saving.saved_model.load.BERTModel which is not the same!
    NOTE: The model can only be saved, when completely initialized (when using the saving api.
    For a not further known reason (but empirically tested), saving a subclassed Keras model with a
    custom `train_step()` function throws an error when not fully initialized. In detail, this line
    `loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)` causes the error. Fully
    initialized means in this context that the given metrics and loss(es) in the `model.compile()` call are not
    built but only set. See `init__()` of the LossesContainer object (wrapper for compiled_metrics property):
    https://github.com/keras-team/keras/blob/3cec735c5602a1bd9880b1b5735c5ce64a94eb76/keras/engine/compile_utils.py#L117
    """
    def __init__(self,
                 encoder: networks.BertEncoder,
                 name: str = "bert",
                 **kwargs):

        super(BERTModel, self).__init__(name=name, **kwargs)

        inputs = copy.copy(encoder.inputs)

        self._config = {
            "encoder": encoder,
            "name": name,
        }

        self.encoder = encoder
        self.inputs = inputs
        self.vocab_size = encoder.get_config()["vocab_size"]

    @property
    def code(self):
        return "bert4rec"

    def call(self, inputs, training: bool = False):
        if isinstance(inputs, list):
            logging.warning('List inputs to the Bert Model are discouraged.')
            inputs = dict([
                (ref.name, tensor) for ref, tensor in zip(self.inputs, inputs)
            ])

        outputs = dict()
        encoder_inputs = {
            "input_word_ids": inputs["input_word_ids"],
            "input_type_ids": inputs["input_type_ids"],
            "input_mask": inputs["input_mask"],
        }
        encoder_network_outputs = self.encoder(encoder_inputs)
        if isinstance(encoder_network_outputs, list):
            outputs['pooled_output'] = encoder_network_outputs[1]
            # When `encoder_network` was instantiated with return_all_encoder_outputs
            # set to True, `encoder_network_outputs[0]` is a list containing
            # all transformer layers' output.
            if isinstance(encoder_network_outputs[0], list):
                outputs['encoder_outputs'] = encoder_network_outputs[0]
                outputs['sequence_output'] = encoder_network_outputs[0][-1]
            else:
                outputs['sequence_output'] = encoder_network_outputs[0]
        elif isinstance(encoder_network_outputs, dict):
            outputs = encoder_network_outputs
        else:
            raise ValueError('encoder_network\'s output should be either a list '
                             'or a dict, but got %s' % encoder_network_outputs)

        return outputs

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs):
        """
        Custom train_step function to alter standard training behaviour

        :return:
        """
        y_true = inputs["masked_lm_ids"]

        with tf.GradientTape() as tape:
            encoder_output = self(inputs, training=True)
            sequence_output = encoder_output["sequence_output"]
            masked_token_sequence = tf.gather(sequence_output, inputs["masked_lm_positions"], axis=1, batch_dims=1)
            # logits
            y_pred = tf.linalg.matmul(masked_token_sequence,
                                      self.encoder._embedding_layer.embeddings, transpose_b=True)

            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super(BERTModel, self).get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)


class BERT4RecModelWrapper:
    def __init__(self,
                 bert_model: BERTModel):
        self.bert_model = bert_model
        self._meta_config = dict(
            {
                "model": "BERT4Rec",
                "tokenizer": None,
                "last_trained": None,
                "trained_on_dataset": None
            }
        )

    @classmethod
    def load(cls, save_path: pathlib.Path, mode: int = 0) -> dict:
        """
        Loads and returns all available assets in conjunction with the ml model. Loads at least a saved
        BERT4Rec model. Depending on the configuration might also load tokenizer with learnt vocab
        TODO: enable correct reloading

        :param save_path: Path to the model directory
        :param mode: The mode determines from where the model is loaded. Available modes are 0, 1, 2.
        0 is default and loads the model relative from the project root. 1 loads the model relative
        from the virtual environment and 2 loads it relative from the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: Dict with all loaded assets
        """
        save_path = utils.determine_model_path(save_path, mode)
        loaded_assets = dict()
        loaded_bert = tf.keras.models.load_model(save_path, custom_objects={
            "BERTModel": BERTModel,
            "AdamWeightDecay": optimizers.AdamWeightDecay,

        })
        wrapper = cls(loaded_bert)
        loaded_assets["model"] = wrapper

        try:
            with open(save_path.joinpath(_META_CONFIG_FILE_NAME)) as jf:
                meta_config = json.load(jf)
                wrapper._meta_config = meta_config

                if "tokenizer" in meta_config and meta_config["tokenizer"] is not None:
                    tokenizer = tokenizer_factory.get_tokenizer(meta_config["tokenizer"])
                    tokenizer.import_vocab_from_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
                    loaded_assets["tokenizer"] = tokenizer

        except FileNotFoundError:
            logging.error(f"The meta configuration/information json file (meta_config.json) could not be found "
                          f"in the supposed model directory: {save_path}")

        return loaded_assets

    def save(self, save_path: pathlib.Path, tokenizer: BaseTokenizer = None, mode: int = 0) -> True:
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

        if self.bert_model.compiled_metrics and not self.bert_model.compiled_metrics._built:
            raise RuntimeError("The model can't be saved yet, as it is not fully instantiated and will "
                               "throw an error during saving. See model docs for more information.")

        self.bert_model.save(save_path)

        if tokenizer:
            tokenizer.export_vocab_to_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
            self.update_meta({"tokenizer": tokenizer.code})

        # save meta config to file
        with open(save_path.joinpath(_META_CONFIG_FILE_NAME), "w") as f:
            json.dump(self._meta_config, f, indent=4)

        return True

    def rank(self,
             encoder_input: dict,
             rank_items: list = None,
             sequence_indexes: Union[list, tf.Tensor] = None) -> tuple:
        """
        Ranks a given set of potential items (rank_items) by probability based on the given sequence.
        The items to be ranked should actually be the indexes of the items to gather their embeddings
        from the embeddings table, so basically just their numeric tokens.

        :param encoder_input: The input to the encoder on which the ranking of the rank_items is based on.
        Should be the regular BERT input dictionary
        :param rank_items: Either list of indexes of the items which should be ranked (in case
        sequence_indexes param is not given) or a list of lists with the indexes of the ranking items
        (if sequence_indexes param is given). If this is a list of lists, the length of the primary list
        should be equal to the length of the sequence_indexes as each index in either list corresponds to
        the index of the other list. If none is given, the whole vocabulary (whole embedding table) is ranked.
        :param sequence_indexes: A list of integers determining the output positions
        which should be used to (individually) rank the items (based on the corresponding output
        of the sequence_output from the encoder). If none is given, the items are ranked based on
        the accumulated encoder output (pooled_output). Could legitimately be the masked language model
        (or simply the positions of the masked tokens which equals the 'masked_lm_positions' tensor).
        :return: A tuple with the first value containing the ranked items from highest probability to lowest
        and the second value containing the probabilities (of each item) (not sorted! -> can be cross-referenced
        with the original rank_items list)
        """
        # TODO: save and reload model correctly (i.e. reloading the model into the right class again to make
        # functions available (as e.g. encoder.get_embedding_table()))
        gathered_embeddings = self.bert_model.encoder._embedding_layer.embeddings

        if rank_items is not None:
            gathered_embeddings = tf.gather(self.bert_model.encoder._embedding_layer.embeddings, rank_items)

        encoder_output = self.bert_model(encoder_input)

        probabilities = list()
        rankings = list()
        if sequence_indexes is not None:
            sequence_output = encoder_output["sequence_output"]
            # iterate over the given sequence_indexes: should be in the shape of [batch, num_tokens] with batch = 1
            # since it's a specific task
            batch = 0
            for token_index in sequence_indexes[batch]:
                token_logits = sequence_output[batch, token_index, :]
                vocab_probabilities, ranking = utils.rank_items(token_logits, gathered_embeddings, rank_items)
                probabilities.append(vocab_probabilities)
                rankings.append(ranking)
        else:
            output_logits = encoder_output["pooled_output"][0, :]
            vocab_probabilities, ranking = utils.rank_items(output_logits, gathered_embeddings, rank_items)
            probabilities.append(vocab_probabilities)
            rankings.append(ranking)

        return rankings, probabilities

    def get_meta(self) -> dict:
        return self._meta_config

    def update_meta(self, updated_info: dict) -> True:
        self._meta_config.update(updated_info)
        return True

    def delete_keys_from_meta(self, keys: Union[list, str]) -> True:
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self._meta_config.pop(key, None)

        return True


if __name__ == "__main__":
    dataloader = BERT4RecML1MDataloader()
    dataloader.generate_vocab()
    ds = dataloader.preprocess_dataset()
    for element in ds.take(1):
        example = element
    #print(example)
    tokenizer = dataloader.get_tokenizer()

    bert_encoder = networks.BertEncoder(30522)
    model = BERTModel(bert_encoder)
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    _ = model(model.inputs)
    wrapper = BERT4RecModelWrapper(model)
    embedding_table = model.encoder._embedding_layer.embeddings
    gathered_embeddings = tf.gather(embedding_table, [0, 3, 30519, 30521])

    predictions = model(example)
    #print(predictions)

    # See model docs to understand, why this is necessary
    model.train_step(example)
    save_path = pathlib.Path("my_model")
    wrapper.save(save_path, tokenizer)
    del wrapper, model, bert_encoder
    loaded_assets = BERT4RecModelWrapper.load(save_path)
    loaded_tokenizer = loaded_assets["tokenizer"]
    loaded_wrapper = loaded_assets["model"]

    print(loaded_wrapper.bert_model)
    print(loaded_wrapper.bert_model(example))

    rank_items = [random.randint(0, 30521) for _ in range(5)]
    #rankings, probabilities = loaded_wrapper.rank(example, rank_items, example["masked_lm_positions"])
    print(rank_items)
    #print(rankings)
