#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: ocr_task
Time: 2025/6/13 15:04
"""
import base64
from io import BytesIO

import cv2
import html2text
import numpy as np
from PIL import Image

from app.common.utils import load_config
from app.core.ocr.models.text_ocr import TextOcr
from app.tasks.base_task import BaseTask

TASK_NAME = "ocr"


class OcrTask(BaseTask):
    def __init__(self):
        config = load_config(TASK_NAME)
        model = TextOcr(config)
        super().__init__(model)

    def predict_images(self, img_base64: str):
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        img = img.convert("RGB")
        img_array = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        ocr_result = self.model.predict(img_array)
        return ocr_result

    @staticmethod
    def html_to_markdown(html_content: str):
        if not html_content:
            return ""
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        return converter.handle(html_content)

    @staticmethod
    def to_dchars(text_word, text_word_region) -> list:
        dchars = []
        for i, words in enumerate(text_word):
            word_region = text_word_region[i]
            for j, word in enumerate(words):
                dchars.append(
                    {
                        "height": round(word_region[j][3][1] - word_region[j][0][1], 2),
                        "width": round(word_region[j][2][0] - word_region[j][0][0], 2),
                        "str": word,
                        "y": round(word_region[j][3][1], 2),
                        "x": round(word_region[j][0][0], 2),
                        "accumulated_y": round(word_region[j][3][1], 2),
                    }
                )
        return dchars


ocr_task = OcrTask()
