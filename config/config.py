# -*- coding: utf-8 -*-
"""
    config
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_mean = [0,0,0]
dataset_std = [0,0,0]