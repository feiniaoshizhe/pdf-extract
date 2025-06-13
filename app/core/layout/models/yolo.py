#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: yolo
Time: 2025/6/13 14:27
"""
import os

import cv2
import numpy as np
import torch
from doclayout_yolo import YOLOv10

from app.common.utils import visualize_bbox


class LayoutYOLOv10:
    def __init__(self, config):

        self.id_to_names = {
            0: 'title',
            1: 'plain text',
            2: 'abandon',
            3: 'figure',
            4: 'figure_caption',
            5: 'table',
            6: 'table_caption',
            7: 'table_footnote',
            8: 'isolate_formula',
            9: 'formula_caption'
        }

        self.model = YOLOv10(config["model_path"])
        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        self.device = config.get('device', 'cpu')

        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def predict(self, images: dict, result_path=None):
        """

        :param images:
        :param result_path:
        :return:
        """
        results = []
        for image_id, image in images.items():
            result = self.model.predict(image, iou=self.iou_thres, verbose=False, device=self.device)[0]
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            scores = result.boxes.conf.tolist()
            keys = ["poly", "category_id", "score"]
            layout_results = [dict(zip(keys, values)) for values in zip(boxes, classes, scores)]

            if self.visualize:
                self.result_visualize(image_id, image, boxes, classes, scores, result_path)

            results.append(
                {
                    "image_id": image_id,
                    "layout_results": layout_results
                }
            )
        return results

    def result_visualize(self, image_id, image, boxes, classes, scores, result_path):
        if self.visualize:
            if not os.path.exists(result_path):
                os.makedirs(result_path)

        if self.iou_thres > 0:
            indices = self.nms_func(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),
                                    iou_threshold=self.iou_thres)
            boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
            if len(boxes.shape) == 1:
                boxes = np.expand_dims(boxes, 0)
                scores = np.expand_dims(scores, 0)
                classes = np.expand_dims(classes, 0)

        vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)
        result_name = f"{image_id}_layout.png"
        # Save the visualized result
        cv2.imwrite(os.path.join(result_path, result_name), vis_result)
