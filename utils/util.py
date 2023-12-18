'''
File: util.py
Project: DoctorTwo
File Created: Thursday, 25th May 2023 11:46:48 am
Author: Arthur JIANG (ArthurSJiang@gmail.com)
-----
Last Modified: Thursday, 25th May 2023 5:07:57 pm
Modified By: Arthur JIANG (ArthurSJiang@gmail.com>)
-----
Copyright (c) 2023 YiEr Tech, Inc.
'''


import json
import os
from os.path import isfile, join
from sys import platform
import shutil
import time
from typing import Dict, NoReturn, Union
import yaml

import colorful as cf
from loguru import logger
import pathlib
import pickle
from collections import OrderedDict
import datetime

def reformat_timestamp(timestamp: float):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

def rm_file(file_path: str) -> NoReturn:
    if isfile(file_path):
        os.remove(file_path)
        
def timer(func):
    def wrap(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        show_timer = os.getenv("SHOW_TIMER")
        if show_timer is True or show_timer == "True" or show_timer == "true":
            logger.info(cf.yellow(f"{func} cost {round(end - start, 2)} s"))
        return res

    return wrap


def copy_file(source, destination, force=False):
    if platform == "win32":
        dir = "\\".join(destination.split("\\")[:-1])
    else:
        dir = "/".join(destination.split("/")[:-1])

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    if force:
        rm_file(destination)
    shutil.copyfile(source, destination)


class DottableDict(dict):
    """A wrapper to dictionary to make possible to key as property."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def convert_dottable(natural_dict: dict) -> DottableDict:
    """Convert a dictionary to DottableDict.
    Args:
        natural_dict (dict): Dictionary to convert to DottableDict.
    Returns:
        DottableDict: Dottable object.
    """
    dottable_dict = DottableDict(natural_dict)
    for k, v in natural_dict.items():
        if type(v) is dict:
            v = convert_dottable(v)
            dottable_dict[k] = v
    return dottable_dict


def convert_dict(dottable_dict: DottableDict) -> Dict:
    """Convert a dottable dictionary to a natural dictionary.
    Args:
        dottable_dict (DottableDict): Dottable dictionary.
    Returns:
        dict: Natural dictionary.
    """
    if type(dottable_dict) is DottableDict:
        natural_dict = OrderedDict()
        for k, v in dottable_dict.items():
            if type(v) is DottableDict:
                v = convert_dict(v)
                natural_dict[k] = v
            elif type(v) is list:
                for idx, item in enumerate(v):
                    v[idx] = convert_dict(item)
                natural_dict[k] = v
            else:
                natural_dict[k] = v

        return dict(natural_dict)
    else:
        return dottable_dict


def load_yaml(file_path: str, advanced: bool = False, show: bool = True) -> Union[Dict, DottableDict]:
    """Load yaml helper.

    Args:
        file_path (str): Absolute file path of yaml file.
        advanced (bool, optional): If True, return DottableDict, else, return Dict. Defaults to False.
        show (bool, optional): Show log info or not. Defaults to True.

    Returns:
        Union[Dict, DottableDict]: Loaded Dict or DottableDict object.
    """
    with open(file_path, "r") as fi:
        ori_dict = yaml.load(fi, Loader=yaml.FullLoader)
        if show:
            logger.info(cf.cyan(f"Load file from {file_path}"))
        if advanced:
            return convert_dottable(ori_dict)
        else:
            return ori_dict


def dump_pickle(obj, file_path: str) -> NoReturn:
    logger.info(cf.cyan(f"Dump file to {file_path}"))
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str):
    logger.info(cf.cyan(f"Load bin from {file_path}"))
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_yaml(obj, file_path, sort_keys: bool = False, show: bool = True) -> NoReturn:
    if platform == "win32":
        dir = "\\".join(file_path.split("\\")[:-1])
    else:
        dir = "/".join(file_path.split("/")[:-1])

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.dump(obj, f, sort_keys=sort_keys)
        if show:
            logger.info(cf.cyan(f"Dump file to {file_path}"))


def find_multiple(n: int, k: int) -> int:
    """Find the minimum n_prime which can divided by k."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def load_json(file_path, show: bool = True):
    with open(file_path, "r") as f:
        if show:
            logger.info(cf.cyan(f"Load file from {file_path}"))
        return json.load(f)
