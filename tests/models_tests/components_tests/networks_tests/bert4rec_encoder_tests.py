"""Tests for transformer-based bert encoder network."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
# pylint: disable=g-direct-tensorflow-import

from bert4rec.models.components.networks import bert4rec_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class Bert4RecEncoderTest(keras_parameterized.TestCase):

    def tearDown(self):
        super(Bert4RecEncoderTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy("float32")

    @parameterized.named_parameters(
        ("encoder", bert4rec_encoder.Bert4RecEncoder),
    )
    def test_dict_outputs_network_creation(self, encoder_cls):
        hidden_size = 32
        sequence_length = 21
        # Create a small BertEncoder for testing.
        if encoder_cls is bert4rec_encoder.Bert4RecEncoder:
            kwargs = {}
        else:
            kwargs = dict(dict_outputs=True)
        test_network = encoder_cls(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            **kwargs)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        data = dict_outputs["sequence_output"]
        pooled = dict_outputs["pooled_output"]

        self.assertIsInstance(test_network.transformer_layers, list)
        self.assertLen(test_network.transformer_layers, 3)
        self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # The default output dtype is float32.
        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    @parameterized.named_parameters(
        ("encoder", bert4rec_encoder.Bert4RecEncoder),
    )
    def test_dict_outputs_all_encoder_outputs_network_creation(self, encoder_cls):
        hidden_size = 32
        sequence_length = 21
        # Create a small BertEncoder for testing.
        test_network = encoder_cls(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            dict_outputs=True)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        all_encoder_outputs = dict_outputs["encoder_outputs"]
        pooled = dict_outputs["pooled_output"]

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertLen(all_encoder_outputs, 3)
        for data in all_encoder_outputs:
            self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # The default output dtype is float32.
        self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    @parameterized.named_parameters(
        ("encoder", bert4rec_encoder.Bert4RecEncoder),
    )
    def test_dict_outputs_network_creation_with_float16_dtype(self, encoder_cls):
        hidden_size = 32
        sequence_length = 21
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # Create a small BertEncoder for testing.
        test_network = encoder_cls(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            dict_outputs=True)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        data = dict_outputs["sequence_output"]
        pooled = dict_outputs["pooled_output"]

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # If float_dtype is set to float16, the data output is float32 (from a layer
        # norm) and pool output should be float16.
        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float16, pooled.dtype)

    @parameterized.named_parameters(
        ("all_sequence_encoder", bert4rec_encoder.Bert4RecEncoder, None, 21),
        ("output_range_encoder", bert4rec_encoder.Bert4RecEncoder, 1, 1),
    )
    def test_dict_outputs_network_invocation(
            self, encoder_cls, output_range, out_seq_len):
        hidden_size = 32
        sequence_length = 21
        vocab_size = 57
        num_types = 7
        # Create a small BertEncoder for testing.
        test_network = encoder_cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            output_range=output_range,
            dict_outputs=True)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        data = dict_outputs["sequence_output"]
        pooled = dict_outputs["pooled_output"]

        # Create a models based off of this network:
        model = tf.keras.Model([word_ids, mask], [data, pooled])

        # Invoke the models. We can't validate the output data here (the models is too
        # complex) but this will catch structural runtime errors.
        batch_size = 3
        word_id_data = np.random.randint(
            vocab_size, size=(batch_size, sequence_length))
        mask_data = np.random.randint(2, size=(batch_size, sequence_length))
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[1], out_seq_len)

        # Creates a BertEncoder with max_sequence_length != sequence_length
        max_sequence_length = 128
        test_network = encoder_cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            dict_outputs=True)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        data = dict_outputs["sequence_output"]
        pooled = dict_outputs["pooled_output"]
        model = tf.keras.Model([word_ids, mask], [data, pooled])
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[1], sequence_length)

        # Creates a BertEncoder with embedding_width != hidden_size
        test_network = encoder_cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            embedding_width=16,
            dict_outputs=True)
        dict_outputs = test_network(
            dict(input_word_ids=word_ids, input_mask=mask))
        data = dict_outputs["sequence_output"]
        pooled = dict_outputs["pooled_output"]
        model = tf.keras.Model([word_ids, mask], [data, pooled])
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[-1], hidden_size)
        self.assertTrue(hasattr(test_network, "_embedding_projection"))

    def test_embeddings_as_inputs(self):
        hidden_size = 32
        sequence_length = 21
        # Create a small BertEncoder for testing.
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        test_network.build(
            dict(input_word_ids=word_ids, input_mask=mask))
        embeddings = test_network.get_embedding_layer()(word_ids)
        # Calls with the embeddings.
        dict_outputs = test_network(
            dict(
                input_word_embeddings=embeddings,
                input_mask=mask))
        all_encoder_outputs = dict_outputs["encoder_outputs"]
        pooled = dict_outputs["pooled_output"]

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertLen(all_encoder_outputs, 3)
        for data in all_encoder_outputs:
            self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # The default output dtype is float32.
        self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    def test_serialize_deserialize(self):
        # Create a network object that sets all of its config options.
        kwargs = dict(
            vocab_size=100,
            hidden_size=32,
            num_layers=3,
            num_attention_heads=2,
            max_sequence_length=21,
            type_vocab_size=12,
            inner_dim=1223,
            inner_activation="relu",
            output_dropout=0.05,
            attention_dropout=0.22,
            initializer="glorot_uniform",
            output_range=-1,
            embedding_width=16,
            embedding_layer=None,
            norm_first=False)
        network = bert4rec_encoder.Bert4RecEncoder(**kwargs)

        # Validate that the config can be forced to JSON.
        _ = network.to_json()

        # Tests models saving/loading.
        model_path = self.get_temp_dir() + "/models"
        network.save(model_path)
        _ = tf.keras.models.load_model(model_path)

    def test_network_creation(self):
        hidden_size = 32
        sequence_length = 21
        # Create a small BertEncoder for testing.
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        data, pooled = test_network([word_ids, mask])

        self.assertIsInstance(test_network.transformer_layers, list)
        self.assertLen(test_network.transformer_layers, 3)
        self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # The default output dtype is float32.
        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

        test_network_dict = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            dict_outputs=True)
        # Create the inputs (note that the first dimension is implicit).
        inputs = dict(
            input_word_ids=word_ids, input_mask=mask)
        _ = test_network_dict(inputs)

        test_network_dict.set_weights(test_network.get_weights())
        batch_size = 2
        vocab_size = 100
        num_types = 2
        word_id_data = np.random.randint(
            vocab_size, size=(batch_size, sequence_length))
        mask_data = np.random.randint(2, size=(batch_size, sequence_length))
        list_outputs = test_network([word_id_data, mask_data])
        dict_outputs = test_network_dict(
            dict(
                input_word_ids=word_id_data,
                input_mask=mask_data))
        self.assertAllEqual(list_outputs[0], dict_outputs["sequence_output"])
        self.assertAllEqual(list_outputs[1], dict_outputs["pooled_output"])

    def test_all_encoder_outputs_network_creation(self):
        hidden_size = 32
        sequence_length = 21
        # Create a small BertEncoder for testing.
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            return_all_encoder_outputs=True)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        all_encoder_outputs, pooled = test_network([word_ids, mask])

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertLen(all_encoder_outputs, 3)
        for data in all_encoder_outputs:
            self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # The default output dtype is float32.
        self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    def test_network_creation_with_float16_dtype(self):
        hidden_size = 32
        sequence_length = 21
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # Create a small BertEncoder for testing.
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        data, pooled = test_network([word_ids, mask])

        expected_data_shape = [None, sequence_length, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        # If float_dtype is set to float16, the data output is float32 (from a layer
        # norm) and pool output should be float16.
        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float16, pooled.dtype)

    @parameterized.named_parameters(
        ("all_sequence", None, 21),
        ("output_range", 1, 1),
    )
    def test_network_invocation(self, output_range, out_seq_len):
        hidden_size = 32
        sequence_length = 21
        vocab_size = 57
        num_types = 7
        # Create a small BertEncoder for testing.
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            output_range=output_range)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
        data, pooled = test_network([word_ids, mask])

        # Create a models based off of this network:
        model = tf.keras.Model([word_ids, mask], [data, pooled])

        # Invoke the models. We can't validate the output data here (the models is too
        # complex) but this will catch structural runtime errors.
        batch_size = 3
        word_id_data = np.random.randint(
            vocab_size, size=(batch_size, sequence_length))
        mask_data = np.random.randint(2, size=(batch_size, sequence_length))
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[1], out_seq_len)

        # Creates a BertEncoder with max_sequence_length != sequence_length
        max_sequence_length = 128
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types)
        data, pooled = test_network([word_ids, mask])
        model = tf.keras.Model([word_ids, mask], [data, pooled])
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[1], sequence_length)

        # Creates a BertEncoder with embedding_width != hidden_size
        test_network = bert4rec_encoder.Bert4RecEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            embedding_width=16)
        data, pooled = test_network([word_ids, mask])
        model = tf.keras.Model([word_ids, mask], [data, pooled])
        outputs = model.predict([word_id_data, mask_data])
        self.assertEqual(outputs[0].shape[-1], hidden_size)
        self.assertTrue(hasattr(test_network, "_embedding_projection"))


if __name__ == "__main__":
    tf.test.main()
