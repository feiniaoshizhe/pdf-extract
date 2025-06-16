#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: utils
Time: 2025/6/13 15:13
"""
import logging

import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from app.config.conf import BASE_DIR

logger = logging.getLogger(__name__)


def load_config(config_name: str):
    config_path = BASE_DIR / "config" / f"{config_name}.yaml"
    config = OmegaConf.load(config_path)
    return config


def colormap(N=256, normalized=False):
    """
    Generate the color map.

    Args:
        N (int): Number of labels (default is 256).
        normalized (bool): If True, return colors normalized to [0, 1]. Otherwise, return [0, 255].

    Returns:
        np.ndarray: Color map array of shape (N, 3).
    """

    def bitget(byteval, idx):
        """
        Get the bit value at the specified index.

        Args:
            byteval (int): The byte value.
            idx (int): The index of the bit.

        Returns:
            int: The bit value (0 or 1).
        """
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap.astype(np.float32) / 255.0

    return cmap


def visualize_bbox(image_path, bboxes, classes, scores, id_to_names, alpha=0.3):
    """
    Visualize layout detection results on an image.

    :param image_path:
    :param bboxes: List of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
    :param classes: List of class IDs corresponding to the bounding boxes.
    :param scores:
    :param id_to_names:Dictionary mapping class IDs to class names.
    :param alpha:Transparency factor for the filled color (default is 0.3).
    :return: np.ndarray: Image with visualized layout detection results.
    """
    # Check if image_path is a PIL.Image.Image object
    if isinstance(image_path, Image.Image):
        image = np.array(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    else:
        image = cv2.imread(image_path)

    overlay = image.copy()

    cmap = colormap(N=len(id_to_names), normalized=False)

    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        class_id = int(classes[i])
        class_name = id_to_names[class_id]

        text = class_name + f":{scores[i]:.3f}"

        color = tuple(int(c) for c in cmap[class_id])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the class name with a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


def get_bbox_from_points(points):
    """
    从8个点坐标计算出矩形的边界 [x1, y1, x2, y2]。
    :param points: 8个点坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: [x1, y1, x2, y2] 格式的 bbox
    """
    x_coords = [points[i] for i in range(0, len(points), 2)]  # 获取所有 x 坐标
    y_coords = [points[i] for i in range(1, len(points), 2)]  # 获取所有 y 坐标
    x1, x2 = min(x_coords), max(x_coords)  # 左边界 x1，右边界 x2
    y1, y2 = min(y_coords), max(y_coords)  # 上边界 y1，下边界 y2
    return [x1, y1, x2, y2]


def calculate_area_overlap(bbox1, bbox2, overlap_ratio_threshold=0.8):
    """
    计算两个bbox的面积重叠比例，并检查该重叠面积占较小bbox面积的比例是否超过阈值。
    :param bbox1: 第一个bbox，格式为 [x1, y1, x2, y2]
    :param bbox2: 第二个bbox，格式为 [x1, y1, x2, y2]
    :param overlap_ratio_threshold: 面积重叠占较小面积的阈值（默认80%）
    :return: 返回是否需要删除较小的bbox
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right > x_left and y_bottom > y_top:
        # 计算交集的面积
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
    else:
        # 如果没有重叠区域，返回0
        overlap_area = 0

    # 计算每个bbox的面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 选择较小的面积
    smaller_area = min(area1, area2)

    if overlap_area == 0:
        return False
    # 判断重叠面积占较小面积的比例
    if overlap_area / smaller_area >= overlap_ratio_threshold:
        return True  # 如果重叠面积比例超过阈值，返回 True 表示需要删除较小的 bbox
    else:
        return False  # 否则，不删除


def remove_small_blocks_from_overlaps(result):
    if not result:
        return result

    blocks = result["layout_dets"]
    if len(blocks) == 1:
        result["layout_dets"] = blocks
    blocks_to_remove = []
    logger.info(f"解析到{len(blocks)}个块")
    # 遍历每一对块
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            bbox1 = get_bbox_from_points(blocks[i]["poly"])  # 转换为 [x1, y1, x2, y2]
            bbox2 = get_bbox_from_points(blocks[j]["poly"])  # 转换为 [x1, y1, x2, y2]

            # 判断是否需要删除较小的块
            if calculate_area_overlap(bbox1, bbox2):
                # 计算高度
                height1 = max(bbox1[3], bbox1[1]) - min(bbox1[3], bbox1[1])
                height2 = max(bbox2[3], bbox2[1]) - min(bbox2[3], bbox2[1])

                if height1 < height2:
                    blocks_to_remove.append(i)
                else:
                    blocks_to_remove.append(j)

    # 删除重叠较小的块
    blocks_to_remove = set(blocks_to_remove)  # 去重
    blocks = [block for idx, block in enumerate(blocks) if idx not in blocks_to_remove]
    logger.info(f"删除了{len(blocks_to_remove)}个重叠块")
    # 更新页面的块数据
    result["layout_dets"] = blocks

    return result
