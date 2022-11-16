import tensorflow as tf


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self,
                 pad_token: int = 0,
                 reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
                 name: str = None):
        super().__init__(reduction, name)
        self.pad_token = pad_token
        
    def call(self, y_true, y_pred):
        mask = y_true != self.pad_token
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.boolean_mask(loss, mask)

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


class MaskedAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, pad_token: int = 0):
        super().__init__()
        self.pad_token = pad_token
        self.total = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=2)
        y_true = tf.cast(y_true, y_pred.dtype)
        match = y_true == y_pred

        mask = y_true != self.pad_token

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        self.total = tf.reduce_sum(match)/tf.reduce_sum(mask)

    def result(self):
        return self.total


def masked_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)
    match = y_true == y_pred

    mask = y_true != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)
