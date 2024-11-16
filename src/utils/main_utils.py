import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logging
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    """read yaml file and return ConfigBox object
    Args:
    path_to_yaml (Path): path to yaml file

    Returns:
    ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError as e:
        logging.error(f"yaml file: {path_to_yaml} failed to load")
        raise e
    except Exception as e:
        logging.error(f"yaml file: {path_to_yaml} failed to load")
        raise e
    

@ensure_annotations 
def create_directories(path_to_directories: list, verbose=True):
    """create directories if they do not exist
    
    Args:
    path_to_directories (list): list of paths to directories
    verbose (bool): print message if directory is created
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"directory: {path} created successfully")


@ensure_annotations
def save_json(data: dict, path_to_json: Path):
    """save data to json file

    Args:

    data (Any): data to be saved
    path_to_json (Path): path to json file
    """
    with open(path_to_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    logging.info(f"json file: {path_to_json} saved successfully")


@ensure_annotations
def load_json(path: Path)-> ConfigBox:
    """load json file

    Args:
    path (Path): path to json file

    Returns:
    ConfigBox: loaded json data
    """
    with open(path) as f:
        content = json.load(f)
    
    logging.info(f"Json file loaded successfully: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data:Any, path:Path):
    """save data to binary file

    Args:
    data (Any): data to be saved
    path (Path): path to binary file
    """
    joblib.dump(value=data,filename=path)
    logging.info(f"binary file: {path} saved successfully")

@ensure_annotations
def load_bin(path:Path)->Any:
    """load binary file

    Args:
    path (Path): path to binary file

    Returns:
    Any: loaded binary data
    """
    data = joblib.load(path)
    logging.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path:Path)-> str:
    """get size of file

    Args:
    path (Path): path to file

    Returns:
    str: size of file
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb}KB"

def decodeImage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        

def encodeImage(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    