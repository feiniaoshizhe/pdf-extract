#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: celery_task
Time: 2025/6/16 14:31
"""
from celery import Celery
from kombu import Queue

from app.config.conf import TZ, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, CELERY_BROKER_DB

# Task name
default_layout_task = "default_layout_task"
default_ocr_task = "default_ocr_task"
table_ocr_task = "table_ocr_task"
latex_ocr_task = "latex_ocr_task"


class Config:
    broker_url = (
        f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{CELERY_BROKER_DB}"
    )
    result_backend = broker_url

    task_serializer = "json"
    result_serializer = "json"
    accept_content = ["json"]
    timezone = TZ
    enable_utc = False
    worker_hijack_root_logger = True  # 禁止celery拦截日志配置(使用loguru的任务配置)

    # task config
    # task_ignore_result = True  # 是否全局设置任务忽略结果
    task_track_started = (
        False  # 当任务启动时报告,需要注意此设置和ignore_result相关配置冲突
    )
    task_acks_late = False

    # worker config
    worker_prefetch_multiplier = 1  # 每个worker每次IO所获取的任务数量
    worker_max_tasks_per_child = 1000  # 执行该任务数后销毁重建新进程
    worker_cancel_long_running_tasks_on_connection_loss = False

    # 开启监控
    worker_send_task_events = True
    task_send_sent_event = True

    # broker
    broker_transport_options = {
        "visibility_timeout": 3600 * 24 * 7,  # 7 days
        "max_retries": 0,
    }

    task_queues = (
        Queue("layout_task_queue"),
        Queue("ocr_task_queue"),
    )

    task_routes = (
        [
            (default_layout_task, {"queue": "layout_task_queue"}),
            (default_ocr_task, {"queue": "ocr_task_queue"}),
            (table_ocr_task, {"queue": "ocr_task_queue"}),
            (latex_ocr_task, {"queue": "ocr_task_queue"}),
        ]
    )

    beat_schedule = {}


celery_app = Celery()
celery_app.config_from_object(Config)
celery_app.set_default()
