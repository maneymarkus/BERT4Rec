"""Tests for Keras-based transformer block layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
# pylint: disable=g-direct-tensorflow-import

from bert4rec.models.components.layers import TransformerEncoderBlock


@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(
    ('base', TransformerEncoderBlock))
class TransformerEncoderBlockLayerTest(keras_parameterized.TestCase):

    def tearDown(self):
        super(TransformerEncoderBlockLayerTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy('float32')

    def test_layer_creation(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        output_tensor = test_layer(data_tensor)
        # The default output of a transformer layer should be the same as the input.
        self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    def test_layer_creation_with_mask(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        # Create a 2-dimensional input (the first dimension is implicit).
        mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
        output_tensor = test_layer([data_tensor, mask_tensor])
        # The default output of a transformer layer should be the same as the input.
        self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    def test_layer_invocation(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        output_tensor = test_layer(data_tensor)

        # Create a models from the test layer.
        model = tf.keras.Model(data_tensor, output_tensor)

        # Invoke the models on test data. We can't validate the output data itself
        # (the NN is too complex) but this will rule out structural runtime errors.
        batch_size = 6
        input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, width))
        _ = model.predict(input_data)

    def test_layer_invocation_with_mask(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        # Create a 2-dimensional input (the first dimension is implicit).
        mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
        output_tensor = test_layer([data_tensor, mask_tensor])

        # Create a models from the test layer.
        model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

        # Invoke the models on test data. We can't validate the output data itself
        # (the NN is too complex) but this will rule out structural runtime errors.
        batch_size = 6
        input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, width))
        # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
        # which here is (batch, sequence_length, sequence_length)
        mask_data = np.random.randint(
            2, size=(batch_size, sequence_length, sequence_length))
        _ = model.predict([input_data, mask_data])

    def test_layer_output_range(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80

        batch_size = 6
        input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, width))
        mask_data = np.random.randint(
            2, size=(batch_size, sequence_length, sequence_length))
        output_tensor = test_layer([input_data, mask_data])

        # The layer only attends to the first token and outputs the first token
        # embedding.
        new_layer = transformer_cls(
            num_attention_heads=10,
            inner_dim=2048,
            inner_activation='relu',
            output_range=1)
        _ = new_layer([input_data, mask_data])
        new_layer.set_weights(test_layer.get_weights())
        new_output_tensor = new_layer([input_data, mask_data])
        self.assertAllClose(
            new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

    def test_layer_output_range_without_mask(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048,
            inner_activation='relu', norm_first=True)
        sequence_length = 21
        width = 80

        batch_size = 6
        input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, width))
        output_tensor = test_layer(input_data)

        # The layer only attends to the first token and outputs the first token
        # embedding.
        new_layer = transformer_cls(
            num_attention_heads=10,
            inner_dim=2048,
            inner_activation='relu',
            output_range=1,
            norm_first=True)
        _ = new_layer(input_data)
        new_layer.set_weights(test_layer.get_weights())
        new_output_tensor = new_layer(input_data)
        self.assertAllClose(
            new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

    def test_layer_output_range_with_pre_norm(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048,
            inner_activation='relu', norm_first=True)
        sequence_length = 21
        width = 80

        batch_size = 6
        input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, width))
        mask_data = np.random.randint(
            2, size=(batch_size, sequence_length, sequence_length))
        output_tensor = test_layer([input_data, mask_data])

        # The layer only attends to the first token and outputs the first token
        # embedding.
        new_layer = transformer_cls(
            num_attention_heads=10,
            inner_dim=2048,
            inner_activation='relu',
            output_range=1,
            norm_first=True)
        _ = new_layer([input_data, mask_data])
        new_layer.set_weights(test_layer.get_weights())
        new_output_tensor = new_layer([input_data, mask_data])
        self.assertAllClose(
            new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

    def test_layer_invocation_with_float16_dtype(self, transformer_cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        test_layer = transformer_cls(
            num_attention_heads=10, inner_dim=2048, inner_activation='relu')
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        # Create a 2-dimensional input (the first dimension is implicit).
        mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
        output_tensor = test_layer([data_tensor, mask_tensor])

        # Create a models from the test layer.
        model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

        # Invoke the models on test data. We can't validate the output data itself
        # (the NN is too complex) but this will rule out structural runtime errors.
        batch_size = 6
        input_data = (10 * np.random.random_sample(
            (batch_size, sequence_length, width)))
        # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
        # which here is (batch, sequence_length, sequence_length)
        mask_data = np.random.randint(
            2, size=(batch_size, sequence_length, sequence_length))
        _ = model.predict([input_data, mask_data])

    def test_transform_with_initializer(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10,
            inner_dim=2048,
            inner_activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        sequence_length = 21
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(sequence_length, width))
        output = test_layer(data_tensor)
        # The default output of a transformer layer should be the same as the input.
        self.assertEqual(data_tensor.shape.as_list(), output.shape.as_list())

    def test_dynamic_layer_sequence(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10,
            inner_dim=2048,
            inner_activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        # Create a 3-dimensional input (the first dimension is implicit).
        width = 30
        input_tensor = tf.keras.Input(shape=(None, width))
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)

        input_length = 17
        input_data = np.ones((1, input_length, width))
        output_data = model.predict(input_data)

        self.assertAllEqual([1, input_length, width], output_data.shape)

    def test_separate_qkv(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=2,
            inner_dim=128,
            inner_activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        # Forward path.
        q_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 16], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        inputs = [q_tensor, kv_tensor, dummy_mask]
        output = test_layer(inputs)
        self.assertEqual(output.shape, q_tensor.shape)


@keras_parameterized.run_all_keras_modes
class TransformerEncoderBlockLayerTestWithoutParams(
    keras_parameterized.TestCase):

    def tearDown(self):
        super(TransformerEncoderBlockLayerTestWithoutParams, self).tearDown()
        tf.keras.mixed_precision.set_global_policy('float32')

    def test_raises_invalid_arg_error_when_q_kv_dims_are_different(self):
        test_layer = TransformerEncoderBlock(
            num_attention_heads=2,
            inner_dim=128,
            inner_activation='relu',
            norm_first=True)
        # Forward path.
        q_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 32], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        inputs = [q_tensor, kv_tensor, dummy_mask]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            test_layer(inputs)

    @parameterized.named_parameters(
        ('output_range_not_none', 2),
        ('output_range_none', None))
    def test_needs_diff_q_kv_att_layer_norm_to_be_true_for_diff_q_and_kv_dims(
            self,
            output_range):
        test_layer = TransformerEncoderBlock(
            num_attention_heads=2,
            inner_dim=128,
            inner_activation='relu',
            output_range=output_range,
            norm_first=True)
        # Forward path.
        q_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 32], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        inputs = [q_tensor, kv_tensor, dummy_mask]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            test_layer(inputs)

        test_layer = TransformerEncoderBlock(
            num_attention_heads=2,
            inner_dim=128,
            inner_activation='relu',
            diff_q_kv_att_layer_norm=True,
            norm_first=True)
        # Forward path.
        test_layer(inputs)

    @parameterized.named_parameters(
        ('norm_first_is_true', True),
        ('norm_first_is_false', False))
    def test_use_query_residual_false_removes_add_op(self, norm_first):
        graph_with_res = tf.Graph()
        with graph_with_res.as_default():
            layer = TransformerEncoderBlock(
                num_attention_heads=2,
                inner_dim=128,
                inner_activation='relu',
                norm_first=norm_first)
            inputs = tf.keras.Input(shape=(None, None, 2))
            outputs = layer(inputs)
            tf.keras.Model(inputs=inputs, outputs=outputs)

        graph_without_res = tf.Graph()
        with graph_without_res.as_default():
            layer = TransformerEncoderBlock(
                num_attention_heads=2,
                inner_dim=128,
                inner_activation='relu',
                norm_first=norm_first,
                use_query_residual=False)
            inputs = tf.keras.Input(shape=(None, None, 2))
            outputs = layer(inputs)
            tf.keras.Model(inputs=inputs, outputs=outputs)
        graph_with_res_names = {x.name for x in graph_with_res.get_operations()}
        graph_without_res_names = {
            x.name for x in graph_without_res.get_operations()
        }

        self.assertIn('transformer_encoder_block/add',
                      list(graph_with_res_names - graph_without_res_names)[0])
        self.assertEmpty(graph_without_res_names - graph_with_res_names)

    @parameterized.named_parameters(
        ('key_dim_is_none', None, 128, 2, 128 // 2),
        ('key_dim_is_not_none', 30, 128, 2, 30))
    def test_key_dim(
            self,
            key_dim,
            q_tensor_last_dim,
            some_num_attention_heads,
            expected):
        some_inner_dim = 32
        some_inner_activation = 'relu'
        test_layer = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            key_dim=key_dim)

        q_tensor = tf.zeros([2, 4, q_tensor_last_dim], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 32], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        test_layer([q_tensor, kv_tensor, dummy_mask])

        self.assertEqual(
            expected,
            test_layer._attention_layer.get_config()['key_dim'])

    @parameterized.named_parameters(
        ('output_last_dim_is_none_use_query_residual_false',
         False,
         None,
         128,
         128),
        ('output_last_dim_is_none_use_query_residual_true',
         True,
         None,
         128,
         128),
        ('output_last_dim_is_not_none', False, 30, 128, 30))
    def test_output_last_dim(
            self,
            use_query_residual,
            output_last_dim,
            q_tensor_last_dim,
            expected):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        test_layer = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            # Must be false for multi-head output to be different from
            # first input's last dim
            use_query_residual=use_query_residual,
            output_last_dim=output_last_dim)

        q_tensor = tf.zeros([2, 4, q_tensor_last_dim], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 32], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        output = test_layer([q_tensor, kv_tensor, dummy_mask])

        self.assertEqual(output.numpy().shape[-1], expected)

    @parameterized.named_parameters(
        ('value_dim_is_none', None, 128, 2, 128 // 2),
        ('value_dim_is_not_none', 30, 128, 2, 30))
    def test_value_dim(
            self,
            value_dim,
            q_tensor_last_dim,
            some_num_attention_heads,
            expected):
        some_inner_dim = 32
        some_inner_activation = 'relu'
        test_layer = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            value_dim=value_dim)

        q_tensor = tf.zeros([2, 4, q_tensor_last_dim], dtype=tf.float32)
        kv_tensor = tf.zeros([2, 8, 32], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
        test_layer([q_tensor, kv_tensor, dummy_mask])

        self.assertEqual(
            expected,
            test_layer._attention_layer.get_config()['value_dim'])


@keras_parameterized.run_all_keras_modes
class TransformerArgumentTest(keras_parameterized.TestCase):

    def test_use_bias_norm_first(self):
        num_attention_heads = 2
        hidden_size = 16
        encoder_block = TransformerEncoderBlock(
            num_attention_heads=num_attention_heads,
            inner_dim=32,
            inner_activation='relu',
            output_dropout=0.1,
            attention_dropout=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6,
            inner_dropout=0.1,
            attention_initializer=tf.keras.initializers.RandomUniform(
                minval=0., maxval=1.))
        # Forward path.
        dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
        inputs = [dummy_tensor, dummy_mask]
        output = encoder_block(inputs)
        self.assertEqual(output.shape, (2, 4, hidden_size))

    def test_norm_first_false_and_diff_q_kv_att_layer_norm_true_raises(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        with self.assertRaises(ValueError):
            TransformerEncoderBlock(
                num_attention_heads=some_num_attention_heads,
                inner_dim=some_inner_dim,
                inner_activation=some_inner_activation,
                norm_first=False,
                diff_q_kv_att_layer_norm=True)

    def test_diff_q_kv_att_layer_norm_is_part_of_config_1(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            norm_first=False)
        self.assertIn('diff_q_kv_att_layer_norm', encoder.get_config())
        self.assertFalse(encoder.get_config()['diff_q_kv_att_layer_norm'])

    def test_diff_q_kv_att_layer_norm_is_part_of_config_2(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            norm_first=True,
            diff_q_kv_att_layer_norm=True)
        self.assertIn('diff_q_kv_att_layer_norm', encoder.get_config())
        self.assertTrue(encoder.get_config()['diff_q_kv_att_layer_norm'])

    def test_use_query_residual_is_part_of_config_1(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation)
        self.assertIn('use_query_residual', encoder.get_config())
        self.assertTrue(encoder.get_config()['use_query_residual'])

    def test_use_query_residual_is_part_of_config_2(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            use_query_residual=False)
        self.assertIn('use_query_residual', encoder.get_config())
        self.assertFalse(encoder.get_config()['use_query_residual'])

    def test_key_dim_is_part_of_config_1(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation)
        self.assertIn('key_dim', encoder.get_config())
        self.assertIsNone(encoder.get_config()['key_dim'])

    def test_key_dim_is_part_of_config_2(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        key_dim = 10
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            key_dim=key_dim)
        self.assertIn('key_dim', encoder.get_config())
        self.assertEqual(key_dim, encoder.get_config()['key_dim'])

    def test_value_dim_is_part_of_config_1(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation)
        self.assertIn('value_dim', encoder.get_config())
        self.assertIsNone(encoder.get_config()['value_dim'])

    def test_value_dim_is_part_of_config_2(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        value_dim = 10
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            value_dim=value_dim)
        self.assertIn('value_dim', encoder.get_config())
        self.assertEqual(value_dim, encoder.get_config()['value_dim'])

    def test_output_last_dim_is_part_of_config_1(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation)
        self.assertIn('output_last_dim', encoder.get_config())
        self.assertIsNone(encoder.get_config()['output_last_dim'])

    def test_output_last_dim_is_part_of_config_2(self):
        some_num_attention_heads = 2
        some_inner_dim = 32
        some_inner_activation = 'relu'
        output_last_dim = 10
        encoder = TransformerEncoderBlock(
            num_attention_heads=some_num_attention_heads,
            inner_dim=some_inner_dim,
            inner_activation=some_inner_activation,
            output_last_dim=output_last_dim)
        self.assertIn('output_last_dim', encoder.get_config())
        self.assertEqual(output_last_dim, encoder.get_config()['output_last_dim'])

    def test_get_config(self):
        num_attention_heads = 2
        encoder_block = TransformerEncoderBlock(
            num_attention_heads=num_attention_heads,
            inner_dim=32,
            inner_activation='relu',
            output_dropout=0.1,
            attention_dropout=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6,
            inner_dropout=0.1,
            attention_initializer=tf.keras.initializers.RandomUniform(
                minval=0., maxval=1.),
            use_query_residual=False,
            key_dim=20,
            value_dim=30,
            output_last_dim=40,
            diff_q_kv_att_layer_norm=True)
        encoder_block_config = encoder_block.get_config()
        new_encoder_block = TransformerEncoderBlock.from_config(
            encoder_block_config)
        self.assertEqual(encoder_block_config, new_encoder_block.get_config())

    @parameterized.parameters({'attention_axes': None}, {'attention_axes': [1]},
                              {'attention_axes': [2]}, {'attention_axes': [1, 2]})
    def test_several_attention_axes(self, attention_axes):
        test_layer = TransformerEncoderBlock(
            inner_dim=32,
            inner_activation='relu',
            output_dropout=0.1,
            attention_dropout=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6,
            inner_dropout=0.1,
            num_attention_heads=10,
            attention_axes=attention_axes)
        num_rows = 21
        num_cols = 13
        width = 80
        # Create a 3-dimensional input (the first dimension is implicit).
        data_tensor = tf.keras.Input(shape=(num_rows, num_cols, width))
        output_tensor = test_layer(data_tensor)
        # The default output of a transformer layer should be the same as the input.
        self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())


if __name__ == '__main__':
    tf.test.main()
