import os
from box.exceptions import BoxError
import yaml
from src.textSummerizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import json
import joblib


@ensure_annotations
def read_yaml(path_to_yaml:Path)  -> ConfigBox:
    
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file :{path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxError:
        raise ValueError("yaML file empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories:list,verbose =True):
    
    
    
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path ,data:dict):
    """
    Saves the given data (a dictionary) as a JSON file at the specified path.

    Args:
        path (Path): The location where the JSON file will be saved.
        data (dict): The dictionary data to be saved in JSON format.

    Returns:
        None
    """
    
    
    with open(path,"w") as f:
        json.dump(data,f,indent= 4)
        
        logger.info(f"json file saved at :{path}")
        

@ensure_annotations
def load_json(path:Path)->ConfigBox:
    
    
    
    with open(path)as f:
        content = json.load(f)
        
    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)
    

@ensure_annotations
def save_bin(data:Any,path :Path):
    
    joblib.dump(value=data ,filename= path)
    logger.info(f"binary file saved at:{path}")


@ensure_annotations
def load_bin(path:Path)-> Any:
    
    
    data =joblib.load(path)
    logger.info(f"Binary File loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Calculate file size and return it as a string in KB."""
    size_in_bytes = os.path.getsize(path)  # Get size in bytes
    size_in_kb = round(size_in_bytes / 1024)  # Convert bytes to kilobytes
    return f"~{size_in_kb} KB"
