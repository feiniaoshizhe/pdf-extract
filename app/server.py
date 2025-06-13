#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: server
Time: 2025/6/13 14:11
"""
from fastapi import FastAPI

app = FastAPI()

app.include_router()
