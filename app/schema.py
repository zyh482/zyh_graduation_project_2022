# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/4/16 10:32 PM
# LAST MODIFIED ON:
# AIM:
from pydantic import BaseModel, Field


# 请求
class Request(BaseModel):
    sentence: str


# 回答
class Response(BaseModel):
    sentence: str
