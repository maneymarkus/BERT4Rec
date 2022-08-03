import os
import pathlib


###################################################
# Constants and Paths                             #
###################################################

VIRTUAL_ENV_PATH = pathlib.Path(os.environ["VIRTUAL_ENV"])
DEFAULT_MODEL_SAVE_PATH = pathlib.Path("saved_models")


def get_virtual_env_path() -> pathlib.Path:
    return VIRTUAL_ENV_PATH


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


def get_default_model_save_path() -> pathlib.Path:
    return DEFAULT_MODEL_SAVE_PATH


if __name__ == "__main__":
    print(get_virtual_env_path())
    print(get_project_root())
    print(get_default_model_save_path())
