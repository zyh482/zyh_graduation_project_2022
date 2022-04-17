# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/4/17 3:23 PM
# LAST MODIFIED ON:
# AIM:
from fastapi import FastAPI
from loguru import logger
import time


def register_echo(app):
    '''register app echo apis 注册心跳和根目录

    Args:
        app: FastAPI Instance

    Returns:
        app: registered FastAPI Instance
    '''

    @app.get('/health')
    def healty():
        return 'ok'

    @app.get('/')
    def hello():
        return f'Hello FastAPI {time.ctime()}'

    @app.get('/err/{sleep}')
    def sleep(sleep):
        logger.info(sleep)
        time.sleep(int(sleep))
        logger.info(f"{sleep} ok...")
        return sleep

    return app

