#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: layout_task
Time: 2025/6/13 15:04
"""
import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from app.common.utils import load_config
from app.core.layout.models.yolo import LayoutYOLOv10
from app.tasks.base_task import BaseTask

TASK_NAME = "layout_detection"

logger = logging.getLogger(__name__)


class LayoutTask(BaseTask):
    def __init__(self):
        config = load_config(TASK_NAME)
        model = LayoutYOLOv10(config)
        super().__init__(model)

    def predict_page_image(self, img_base64: str):
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        img = img.convert("RGB")
        img_array = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        layout_result = self.model.predict(img_array)
        return layout_result

    @staticmethod
    def sort_layout_dets(layout_dets, sort_by="top_left_y_then_x"):
        """
        对 layout_dets 中的元素按照矩形框坐标排序

        参数:
            layout_dets: 包含 poly 和 category_id 的字典列表
            sort_by: 排序方式，可选:
                - 'top_left_x': 按左上角 x 坐标 (x0) 排序
                - 'top_left_y': 按左上角 y 坐标 (y0) 排序
                - 'top_left_y_then_x': 先按 y0 排序，再按 x0 排序（默认）

        返回:
            排序后的 layout_dets
        """
        if sort_by == "top_left_x":
            # 按左上角 x 坐标 (x0) 排序
            return sorted(layout_dets, key=lambda item: item["poly"][0])
        elif sort_by == "top_left_y":
            # 按左上角 y 坐标 (y0) 排序
            return sorted(layout_dets, key=lambda item: item["poly"][1])
        elif sort_by == "top_left_y_then_x":
            # 先按 y0 排序，再按 x0 排序（类似阅读顺序）
            return sorted(
                layout_dets, key=lambda item: (item["poly"][1], item["poly"][0])
            )
        else:
            return layout_dets


layout_task = LayoutTask()
