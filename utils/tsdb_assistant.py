# -*- coding:utf-8 -*-
"""
File: tsdb_assistant.py
Project: DoctorTwo
File Created: Sunday, 18th June 2023 6:42:43 pm
Author: Arthur JIANG (ArthurSJiang@gmail.com)
-----
Last Modified: Monday, 19th June 2023 4:41:33 pm
Modified By: Arthur JIANG (ArthurSJiang@gmail.com>)
-----
Copyright (c) 2023 YiEr Tech, Inc.
-----
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
"""


import colorful as cf
from datetime import datetime, timedelta
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb import InfluxDBClient

from loguru import logger

from tqdm import tqdm
from utils.util import load_yaml, timer
from typing import NoReturn
import json


class InfluxDBHelper(object):
    def __init__(self,
                 config_file: str = "tsdb_config.yml",
                 exp_name: str = None,
                 hparams: dict = None):
        self._config = load_yaml(config_file, advanced=True)
        logger.info(self._config.influxdb)
        self._client = InfluxDBClient(host=self._config.influxdb.host,
                                      port=self._config.influxdb.port,
                                      username=self._config.influxdb.username,
                                      password=self._config.influxdb.password)
        self._dashboard_url = self._config.chronograf.dashboard_url
        self._exp_name = exp_name
        self._hparams = hparams

        self._start_time = datetime.now()
        self._lower_time = self._start_time.strftime(
            "%Y-%m-%dT%H %M").replace(" ", "%3A") + "%3A00.000Z"
        self._upload_hparams()

    @timer
    def _upload_hparams(self) -> NoReturn:
        hparams_db = "hparams"

        if hparams_db not in self._client.get_list_database():
            self._client.create_database(hparams_db)

        logger.info("Upload hparams start.")
        point = {
            "measurement": "hparams",
            "tags": {
                "exp_name": self._exp_name
            },
            "fields": self._hparams
        }
        logger.info(cf.cyan(json.dumps(point["fields"], indent=4)))

        self._client.write_points([point], database=hparams_db)

        logger.info("Upload hparams end.")
        logger.info(f"{cf.cyan('Dashboard: ')}{self._dashboard_url}&lower={self._lower_time}")

    @timer
    def track_metrics(self, metrics: dict = None, subset: str = "train") -> NoReturn:
        metrics_db = "metrics"
        if metrics_db not in self._client.get_list_database():
            self._client.create_database(metrics_db)

        logger.info("Upload metrics start.")
        point = {
            "measurement": "metrics",
            "tags": {
                "exp_name": self._exp_name,
                "subset": subset
            },
            "fields": metrics
        }
        self._client.write_points([point], database=metrics_db)
        logger.info("Upload metrics end.")
