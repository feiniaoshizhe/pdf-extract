#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: test
Time: 2025/6/16 11:37
"""
from paddleocr.ppstructure.utility import init_args

parser = init_args()
args = parser.parse_args([])
print(args)
