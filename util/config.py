import glob
import json
import os
import data
from typing import Dict, Any, List


def get_default_configs(path: str = 'config') -> Dict[str, Dict[str, Any]]:
    """
    Returns default configs for the datasets and training objectives located in the given directory path. Each dataset
    has its own subdirectory containing .json files for the supervised and barlow training objectives. These .json files
    contain the default input parameters.

    :param path: path to the config directory. Defaults to 'config'
    :return: Dictionary with default input parameters for each dataset and training objectives.
    """

    return {
        dataset: {
            data.get_filename(f): parse_config(f) for f in glob.glob(os.path.join(path, dataset, '*.json'))
        } for dataset in os.listdir(path)
    }


def parse_config(file_path: str) -> Dict[str, Any]:
    """
    Opens and loads a .json file

    :param file_path: path to the .json file
    :return:
    """

    with open(file_path) as config_file:
        return json.load(config_file)


def config_to_args(config: Dict[str, Any]) -> List[str]:
    """
    Converts a dictionary to a list of arguments that can be processed by argparse.

    :param config: dictionary containing the input arguments
    :return: arguments as list of string that can be processed by argparse
    """
    return [element for element in [[f"--{key}", str(value)] for key, value in config.items()] for element in element]
