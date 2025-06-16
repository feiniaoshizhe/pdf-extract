#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: tasks
Time: 2025/6/13 14:11
"""
from app.common.utils import remove_small_blocks_from_overlaps
from app.core.engine.celery_task import (
    celery_app,
    default_layout_task,
    default_ocr_task,
    latex_ocr_task,
    table_ocr_task)
from app.tasks.layout_task import layout_task
from app.tasks.ocr_task import ocr_task


@celery_app.task(name=default_layout_task, ignore_result=False)
def default_layout_task(image_id: str, img_base64: str):
    layout_results = layout_task.predict_page_image(img_base64)
    layout_results = remove_small_blocks_from_overlaps(layout_results)
    layout_results["layout_dets"] = layout_task.sort_layout_dets(
        layout_results["layout_dets"]
    )
    result = {f"{image_id}": layout_results}
    return result


@celery_app.task(name=default_ocr_task, ignore_result=False)
def default_ocr_parse_task(image_id: str, img_base64: str):
    ocr_results = ocr_task.predict_images(image_id, img_base64)
    text = ocr_results["text"]
    text_word = ocr_results["text_word"]
    text_word_region = ocr_results["text_word_region"]
    bbox = ocr_results.get("text_region", []) or ocr_results.get("cell_bbox", []) or None
    # 解析每个字符
    dchars = ocr_task.to_dchars(text_word, text_word_region)
    # 表格转换成html形式
    table_markdown = ocr_task.html_to_markdown(ocr_results.get("tabel_html", ""))

    result = {
        f"{image_id}": {
            "text": text,
            "bbox": bbox,
            "chars": dchars,
            "table_markdown": table_markdown,
        }
    }
    return result


@celery_app.task(name=latex_ocr_task, ignore_result=False)
def latex_ocr_task(image_id: str, img_base64: str):
    pass


@celery_app.task(name=table_ocr_task, ignore_result=False)
def table_ocr_task(image_id: str, img_base64: str):
    pass
