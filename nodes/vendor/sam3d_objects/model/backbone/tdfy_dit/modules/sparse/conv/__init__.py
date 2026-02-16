# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from .. import BACKEND

log = logging.getLogger("sam3dobjects")

SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'


def __from_env():
    import os

    global SPCONV_ALGO
    env_spconv_algo = os.environ.get("SPCONV_ALGO")
    if env_spconv_algo is not None and env_spconv_algo in [
        "auto",
        "implicit_gemm",
        "native",
    ]:
        SPCONV_ALGO = env_spconv_algo
    log.info("spconv algo: %s", SPCONV_ALGO)


__from_env()

if BACKEND == "torchsparse":
    from .conv_torchsparse import *
elif BACKEND == "spconv":
    from .conv_spconv import *
