#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: __init__
Time: 2025/6/13 16:34
"""
from fastapi import APIRouter

from app.api.routes import layout

base_router = APIRouter()
base_router.include_router(layout.router)
