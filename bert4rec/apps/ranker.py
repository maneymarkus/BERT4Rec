import numpy as np
import tensorflow as tf

from bert4rec import dataloaders


class Ranker(tf.Module):
    """
    This example app gets an unprocessed sequence of items and a rank_item and returns the rank
    of this specific item relative to the whole vocabulary
    """
    def __init__(self, ranker_model: tf.keras.Model, dataloader: dataloaders.BaseDataloader = None):
        super().__init__()
        self.ranker_model = ranker_model
        if dataloader is None:
            dataloader = dataloaders.BERT4RecDataloader(max_seq_len=512, max_predictions_per_seq=0)
        self.dataloader = dataloader

    def __call__(self, sequence: list[str], rank_item: str, rank_items: list[str] = None):
        model_input = self.dataloader.prepare_inference(sequence)

        predictions = self.ranker_model(model_input, training=False)

        sequence_output = predictions["sequence_output"]
        # get output for the last token in the sequence (the manually added masked token to trigger inference)
        prediction_logits = sequence_output[:, -1, :]
        embedding_table = self.ranker_model.encoder.get_embedding_table()

        if rank_items is not None:
            tokenized_rank_items = self.dataloader.tokenizer.tokenize(rank_items)
            embedding_table = tf.gather(embedding_table, tokenized_rank_items)

        # multiply encoder sequence output with transposed embedding table to get vocab logits (pre-probabilities
        # of the vocab)
        vocab_logits = tf.linalg.matmul(prediction_logits, embedding_table, transpose_b=True)

        # if the ranker models has a prediction mask, apply it to the vocab logits to prevent
        # unwanted tokens from being predicted but only when the whole vocabulary is ranked
        if rank_items is None and hasattr(self.ranker_model, "prediction_mask") \
                and self.ranker_model.prediction_mask is not None:
            vocab_logits += self.ranker_model.prediction_mask

        rank_item_token = self.dataloader.tokenizer.tokenize(rank_item)

        # sort the indexes of the vocab according to their probability
        sorted_indexes = tf.argsort(vocab_logits[0], direction="DESCENDING")
        # if the whole embedding table was used the sorted indexes simply are the ranked vocabulary
        vocab_ranking = sorted_indexes
        # but if only a subset of items from the vocabulary should be ranked the respective items
        # need to be gathered from the tokenized_rank_items list
        if rank_items is not None:
            vocab_ranking = tf.gather(tokenized_rank_items, sorted_indexes)

        # np.where() yields tuple and array
        rank = np.where(vocab_ranking.numpy() == rank_item_token)[0][0]

        assert_string = f"Rank of \"{rank_item}\" is {rank} in the given sequence:\n{sequence}\n"
        if rank_items is not None:
            assert_string += f"relative to {len(rank_items)} other elements:\n{rank_items}"
        else:
            assert_string += "relative to the whole vocabulary"

        return rank, assert_string
