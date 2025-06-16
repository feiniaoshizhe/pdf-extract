#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: conf
Time: 2025/6/13 18:23
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TZ = pytz.timezone(os.getenv("TZ", "Asia/Shanghai"))
# [DEVICE]
DEVICE = os.getenv("DEVICE", "cuda")

# [redis]
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_CACHE_DB = int(os.getenv("REDIS_CACHE_DB", 0))
CELERY_BROKER_DB = int(os.getenv("CELERY_BROKER_DB", 1))
