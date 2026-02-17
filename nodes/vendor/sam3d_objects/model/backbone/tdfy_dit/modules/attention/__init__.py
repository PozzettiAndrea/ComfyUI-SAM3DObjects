# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger

DEBUG = False

from .full_attn import *
from .modules import *

from comfy_attn import set_backend, get_backend_label, auto_detect_precision
set_backend("auto")
logger.info(f"SAM3D dense attention: {get_backend_label()}, precision: {auto_detect_precision()}")
