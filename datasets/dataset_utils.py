"""
This file contains basic utility function concerning datasets, like e.g. downloading and unzipping
"""

from absl import logging
from functools import reduce
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import tqdm
import wget
import zipfile


def get_byte_size(path: pathlib.Path) -> int:
    """
    Calculates size of file at given path. In case of directory
    the size is calculated based on its contents

    :param path: path to file/directory
    :return: Size of file
    """
    size = 0
    if path.is_file():
        return os.path.getsize(path)
    elif path.is_dir():
        for f in path.glob("**/*"):
            size += os.path.getsize(f)
        return size
    else:
        return -1


def is_available(path: pathlib.Path, expected_size: int) -> bool:
    """
    Determines if the resource is fully available (/downloaded) on the basis
    of the provided expected_size

    :param path: Path to resource
    :param expected_size: Expected size of the fully available (/downloaded) resource
    :return: The info whether the resource is (fully) available
    """
    if not path.exists():
        return False
    path_size = get_byte_size(path)
    if path_size * 0.98 < expected_size < path_size * 1.02:
        return True
    return False


def download(url: str, save_path: pathlib.Path):
    """
    Download a resource from `url` to `save_path`

    :param url: URL pointing to resource to be downloaded
    :param save_path: Path to download to
    :return: None
    """
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    wget.download(url, str(save_path), bar=custom_progress_bar)
    # Just print a single new line after the download
    print()


def custom_progress_bar(current, total, width=80):
    """
    This function provides a custom progress bar when using the wget.download() functionality
    due to PyCharm having issues displaying the standard progress bar
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def unzip(zip_path: pathlib.Path, save_path: pathlib.Path) -> None:
    """
    Uncompress a zip-compressed directory

    :param zip_path: Path pointing to zip-compressed directory
    :param save_path: Path to extract the zip-directory to
    :return: None
    """
    zip = zipfile.ZipFile(zip_path)
    size = sum([zinfo.file_size for zinfo in zip.filelist])
    tqdm.tqdm(zip.extractall(save_path), total=size)
    zip.close()


def untar(tar_path: pathlib.Path, save_path: pathlib.Path) -> None:
    """
    Uncompress a tar-compressed directory

    :param tar_path: Path pointing to tar-compressed directory
    :param save_path: Path to extract the tar-directory to
    :return: None
    """
    tar = tarfile.open(tar_path)
    size = reduce(lambda x, y: getattr(x, 'size', x) + getattr(y, 'size', y), tar.getmembers())
    tqdm.tqdm(tar.extractall(save_path), total=size)
    tar.close()


def download_and_unpack_to_folder(
        url: str,
        destination: pathlib.Path,
        compression_method: str = None,
        content_is_dir: bool = False):
    """
    Download a resource at a given `url` to `destination` and optionally
    unpack it based on the given `compression_method`

    :param url: URL pointing to resource
    :param destination: Path to download the resource to
    :param compression_method: (Optional) Determines compression method used to minimize resource
    :param content_is_dir: (Optional) Determines if resource is directory
    :return: None
    """
    temp_root = pathlib.Path(tempfile.mkdtemp())
    temp_archive = temp_root.joinpath(os.path.basename(url))
    temp_folder = temp_root.joinpath('folder')
    logging.info("Start downloading...")
    download(url, temp_archive)
    logging.info("Start unpacking...")
    if compression_method is not None:
        if compression_method == "zip":
            unzip(temp_archive, temp_folder)
        elif compression_method == "tar":
            untar(temp_archive, temp_folder)
        else:
            raise ValueError("This compression method is not implemented yet.")
    if content_is_dir:
        temp_folder = temp_folder.joinpath(os.listdir(temp_folder)[0])
    shutil.copytree(temp_folder, destination, dirs_exist_ok=True)
    shutil.rmtree(temp_root)
