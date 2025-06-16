#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: paddle_ocr
Time: 2025/6/13 14:21
"""
import logging
import re

import torch
from paddleocr.ppstructure.utility import init_args
from paddleocr.tools.infer.predict_system import TextSystem

from app.config.conf import DEVICE

logger = logging.getLogger(__name__)

STYLE_TOKEN_RE = "<(?:strike|sup|sub|b|i|overline|underline|\/(?:strike|sup|sub|b|i|overline|underline))>"


class TextOcr:

    def __init__(self, config):
        self.config = config
        args = self._get_parser_args()
        args.return_word_box = True
        self.model = TextSystem(args)

    def _get_parser_args(self):
        logger.info("init ocr text parser args")
        parser = init_args()
        args = parser.parse_args([
            f"--det_model_dir={self.config.det_model_dir}",
            f"--rec_model_dir={self.config.rec_model_dir}",
            f"--rec_char_dict_path={self.config.rec_char_dict_path}",
            "--max_batch_size=1"
            "--rec_batch_num=1"
        ])
        if DEVICE == "cuda" and torch.cuda.is_available():
            args.use_gpu = True
        elif DEVICE.startswith("npu"):
            logger.info("use npu")
            args.use_npu = True
            args.use_gpu = False
        else:
            args.use_gpu = False
        return args

    def predict(self, image):
        filter_boxes, filter_rec_res, time_dict = self.model(image)
        ocr_result = {
            "text": "",
            "bbox": None,
            "chars": [],
            "chars_region": []
        }

        for box, ocr_res in zip(filter_boxes, filter_rec_res):
            text = ocr_res[0]
            score = ocr_res[1]
            text = re.sub(STYLE_TOKEN_RE, "", text)
            char_list, char_box_list = self._get_each_chars_bbox(text=text, box=box, char_info=ocr_res[2])

            ocr_result["text"] += text
            ocr_result["chars"].extend(char_list)
            ocr_result["chars_region"].extend(char_box_list)

        return ocr_result

    @staticmethod
    def _get_each_chars_bbox(text, box, char_info):
        """
        根据OCR识别和检测的结果计算每个词的检测框
        :param text: ocr 识别的文本
        :param box: ocr 识别的检测区块
        :param char_info: 识别的每个字符信息
        :return:
        """

        col_num, word_list, word_col_list, state_list = char_info

        box = box.tolist()
        bbox_x_start = box[0][0]
        bbox_x_end = box[1][0]
        bbox_y_start = box[0][1]
        bbox_y_end = box[2][1]

        cell_width = (bbox_x_end - bbox_x_start) / col_num
        default_char_width = (bbox_x_end - bbox_x_start) / len(text)

        word_box_list = []
        word_box_content_list = []
        for word, word_col, state in zip(word_list, word_col_list, state_list):
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
            else:
                char_width = default_char_width
            for center_idx in word_col:
                center_x = (center_idx + 0.5) * cell_width
                cell_x_start = max(int(center_x - char_width / 2), 0) + bbox_x_start
                cell_x_end = (
                        min(int(center_x + char_width / 2), bbox_x_end - bbox_x_start)
                        + bbox_x_start
                )
                cell = (
                    (cell_x_start, bbox_y_start),
                    (cell_x_end, bbox_y_start),
                    (cell_x_end, bbox_y_end),
                    (cell_x_start, bbox_y_end),
                )
                word_box_list.append(cell)
            word_box_content_list += word

        return word_box_content_list, word_box_list
