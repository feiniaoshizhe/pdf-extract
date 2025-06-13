#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright DataGrand Tech Inc. All Rights Reserved.
Author: youshun xu
File: layout
Time: 2025/6/13 16:34
"""
from fastapi import APIRouter
from pydantic import BaseModel, model_validator

from app.api.routes.schema import LayoutResponse

router = APIRouter(prefix="/layout", tags=["layout"])


class LayoutRequestModel(BaseModel):
    images: dict = None
    pdf_path: str = None

    @model_validator(mode="after")
    def check(self):
        if self.images and self.pdf_path:
            raise ValueError("Only one of 'images' or 'pdf_path' is allowed, not both.")
        if not self.images and not self.pdf_path:
            raise ValueError("")
        return self


@router.post(
    "",
    dependencies=[],
    response_model=LayoutResponse
)
def default_layout(
        body: LayoutRequestModel
):
    if body.pdf_path:
        # TODO download pdf to local temp dir and convert to image
        pass
    if body.images:
        # TODO
        pass
