#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: paddle_ocr
Time: 2025/6/13 14:21
"""
from paddleocr import PaddleOCR


class TestOcr(PaddleOCR):
    def __init__(self, config):
        super().__init__(**config)

    def predict(self, image, **kwargs):
        ppocr_res = self.ocr(image, **kwargs)[0]
        ocr_res = []
        for box_ocr_res in ppocr_res:
            p1, p2, p3, p4 = box_ocr_res[0]
            text, score = box_ocr_res[1]
            pass
