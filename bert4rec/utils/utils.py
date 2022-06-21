import os
import pathlib


###################################################
# Constants and Paths                             #
###################################################

VIRTUAL_ENV_PATH = os.environ["VIRTUAL_ENV"]
DEFAULT_MODEL_SAVE_PATH = pathlib.Path(os.environ["VIRTUAL_ENV"]).joinpath("saved_models")
DEFAULT_MODEL_TRAIN_PATH = pathlib.Path("checkpoints/")


def get_virtual_env_path() -> pathlib.Path:
    return pathlib.Path(VIRTUAL_ENV_PATH)


def get_default_model_save_path() -> pathlib.Path:
    return DEFAULT_MODEL_SAVE_PATH


def get_default_model_train_path() -> pathlib.Path:
    return DEFAULT_MODEL_TRAIN_PATH
