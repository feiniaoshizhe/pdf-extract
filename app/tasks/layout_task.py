#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: layout_task
Time: 2025/6/13 15:04
"""
from app.common.utils import load_config
from app.core.layout.models.yolo import LayoutYOLOv10
from app.tasks.base_task import BaseTask

TASK_NAME = "layout_detection"


class LayoutTask(BaseTask):
    def __init__(self):
        config = load_config(TASK_NAME)
        model = LayoutYOLOv10(config)
        super().__init__(model)

    def predict_images(self):
        pass

    def predict_pdf(self):
        pass
