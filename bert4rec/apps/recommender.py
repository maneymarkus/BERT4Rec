import tensorflow as tf

from bert4rec import dataloaders


class Recommender(tf.Module):
    def __init__(self, recommender_model: tf.keras.Model, dataloader: dataloaders.BaseDataloader = None):
        super().__init__()
        self.recommender_model = recommender_model
        if dataloader is None:
            dataloader = dataloaders.BERT4RecDataloader(max_seq_len=512, max_predictions_per_seq=0)
        self.dataloader = dataloader

    def __call__(self, sequence: list[str]):
        model_input = self.dataloader.prepare_inference(sequence)

        predictions = self.recommender_model(model_input, training=False)

        if "mlm_logits" in predictions:
            # get output for the last token in the sequence of *masked lm logits*
            vocab_logits = predictions["mlm_logits"][:, -1]

        else:

            sequence_output = predictions["sequence_output"]
            # get output for the last token in the sequence (the manually added masked token to trigger inference)
            prediction_logits = sequence_output[:, -1, :]

            embedding_table = self.recommender_model.encoder.get_embedding_table()

            # multiply encoder sequence output with transposed embedding table to get vocab logits (pre-probabilities
            # of the vocab
            vocab_logits = tf.linalg.matmul(prediction_logits, embedding_table, transpose_b=True)

            # if the recommender models has a prediction mask, apply it to the vocab logits to prevent
            # unwanted tokens from being predicted
            if hasattr(self.recommender_model, "prediction_mask") \
                    and self.recommender_model.prediction_mask is not None:
                vocab_logits += self.recommender_model.prediction_mask

        # get most probable vocab index (or simply token)
        vocab_index = tf.argmax(vocab_logits[0])
        # convert token to human-readable recommendation
        recommendation = self.dataloader.tokenizer.detokenize(vocab_index)
        return recommendation
