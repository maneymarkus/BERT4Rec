import tensorflow as tf

from .adam_w_optimizer import *


def create_adam_w_optimizer(init_lr: float = 1e-4,
                            num_train_steps: int = 100000,
                            num_warmup_steps: int = 100,
                            end_lr: float = 0.0,
                            weight_decay_rate: float = 0.01,
                            beta_1: float = 0.9,
                            beta_2: float = 0.999,
                            epsilon: float = 1e-6,
                            exclude_from_weight_decay: list = None) -> tf.keras.optimizers.Adam:
    """
    Create Adam optimizer with weight decay. Initial values from:
    https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/optimization.py
    (except for first three params)

    :param init_lr: Initial learning rate. Default value from:
    https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run_ml-1m.sh#L45
    :param num_train_steps: Number of training steps
    :param num_warmup_steps: Number of warmup steps. Initial value from:
    https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run_ml-1m.sh#L44
    :param end_lr:
    :param weight_decay_rate:
    :param beta_1:
    :param beta_2:
    :param epsilon:
    :param exclude_from_weight_decay:
    :return:
    """

    if exclude_from_weight_decay is None:
        exclude_from_weight_decay = ['LayerNorm', 'layer_norm', 'bias']

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=end_lr)
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps)

    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=exclude_from_weight_decay)

    return optimizer


optimizers_map = {
    "adamw": create_adam_w_optimizer
}


def get(identifier: str = "adamw", **kwargs) -> tf.keras.optimizers.Optimizer:
    """
    Factory method to return a concrete optimizer instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier in optimizers_map:
        return optimizers_map[identifier](**kwargs)
    else:
        raise ValueError(f"{identifier} is an unknown optimizer identifier!")
